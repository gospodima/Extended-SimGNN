import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau

from layers import AttentionModule, TensorNetworkModule, DiffPool
from utils import calculate_ranking_correlation, calculate_prec_at_k, gen_pairs

from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree

import matplotlib.pyplot as plt


class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1))
            
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2))
            
            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3))
            
            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')
        
        if self.args.diffpool:
            self.attention = DiffPool(self.args)
        else:
            self.attention = AttentionModule(self.args)
                      
        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2, batch_1, batch_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for target graphs.
        :param abstract_features_2: Feature matrix for source graphs.
        :param batch_1: Batch vector for source graphs, which assigns each node to a specific example
        :param batch_1: Batch vector for target graphs, which assigns each node to a specific example
        :return hist: Histsogram of similarity scores.
        """
        abstract_features_1, mask_1 = to_dense_batch(abstract_features_1, batch_1)
        abstract_features_2, mask_2 = to_dense_batch(abstract_features_2, batch_2)

        B1, N1, _ = abstract_features_1.size()
        B2, N2, _ = abstract_features_2.size()

        mask_1 = mask_1.view(B1, N1)
        mask_2 = mask_2.view(B2, N2)
        num_nodes = torch.max(mask_1.sum(dim=1), mask_2.sum(dim=1))

        scores = torch.matmul(abstract_features_1, abstract_features_2.permute([0,2,1])).detach()

        hist_list = []
        for i, mat in enumerate(scores):
            mat = torch.sigmoid(mat[:num_nodes[i], :num_nodes[i]]).view(-1)
            hist = torch.histc(mat, bins=self.args.bins)
            hist = hist/torch.sum(hist)
            hist = hist.view(1, -1)
            hist_list.append(hist)
        
        return torch.stack(hist_list).view(-1, self.args.bins)

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = F.relu(features)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_2(features, edge_index)
        features = F.relu(features)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_3(features, edge_index)
        return features
        
    def diffpool(self, abstract_features, edge_index, batch):
        """
        Making differentiable pooling.
        :param abstract_features: Node feature matrix.
        :param edge_index: Edge indices
        :param batch: Batch vector, which assigns each node to a specific example
        :return pooled_features: Graph feature matrix.
        """
        x, mask = to_dense_batch(abstract_features, batch)
        adj = to_dense_adj(edge_index, batch)
        return self.attention(x, adj, mask)

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index
        features_1 = data["g1"].x
        features_2 = data["g2"].x
        batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
        batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        if self.args.histogram:
            hist = self.calculate_histogram(abstract_features_1, abstract_features_2, batch_1, batch_2)
        
        if self.args.diffpool:
            pooled_features_1 = self.diffpool(abstract_features_1, edge_index_1, batch_1)
            pooled_features_2 = self.diffpool(abstract_features_2, edge_index_2, batch_2)
        else:
            pooled_features_1 = self.attention(abstract_features_1, batch_1)
            pooled_features_2 = self.attention(abstract_features_2, batch_2)
            
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        
        if self.args.histogram:
            scores = torch.cat((scores, hist), dim=1)
        
        scores = F.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores)).view(-1)
        return score


class SimGNNTrainer(object):
    """
    SimGNN model trainer.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.process_dataset()
        self.setup_model()

    def setup_model(self):
        """
        Creating a SimGNN.
        """
        self.model = SimGNN(self.args, self.number_of_labels)
        
    def process_dataset(self):
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")

        self.training_graphs = GEDDataset('datasets/{}'.format(self.args.dataset), self.args.dataset, train=True) 
        self.testing_graphs = GEDDataset('datasets/{}'.format(self.args.dataset), self.args.dataset, train=False) 
        self.nged_matrix = self.training_graphs.norm_ged
        self.real_data_size = self.nged_matrix.size(0)
        
        if self.args.synth:
            # self.synth_data_1, self.synth_data_2, _, synth_nged_matrix = gen_synth_data(500, 10, 12, 0.5, 0, 3)
            self.synth_data_1, self.synth_data_2, _, synth_nged_matrix = gen_pairs(self.training_graphs.shuffle()[:500], 0, 3)  
            
            real_data_size = self.nged_matrix.size(0)
            synth_data_size = synth_nged_matrix.size(0)
            self.nged_matrix = torch.cat((self.nged_matrix, torch.full((real_data_size, synth_data_size), float('inf'))), dim=1)
            synth_nged_matrix = torch.cat((torch.full((synth_data_size, real_data_size), float('inf')), synth_nged_matrix), dim=1)
            self.nged_matrix = torch.cat((self.nged_matrix, synth_nged_matrix))
        
        if self.training_graphs[0].x is None:
            max_degree = 0
            for g in self.training_graphs + self.testing_graphs + (self.synth_data_1 + self.synth_data_2 if self.args.synth else []):
                if g.edge_index.size(1) > 0:
                    max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.training_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree
        
        # labeling of synth data according to real data format    
            if self.args.synth:
                for g in self.synth_data_1 + self.synth_data_2:
                    g = one_hot_degree(g)
                    g.i = g.i + real_data_size
        elif self.args.synth:
            for g in self.synth_data_1 + self.synth_data_2:
                g.i = g.i + real_data_size
                # g.x = torch.cat((g.x, torch.zeros((g.x.size(0), self.training_graphs.num_features-1))), dim=1)
                    
        self.number_of_labels = self.training_graphs.num_features

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: Zipped loaders as list.
        """
        if self.args.synth:
            synth_data_ind = random.sample(range(len(self.synth_data_1)), 100)
        
        source_loader = DataLoader(self.training_graphs.shuffle() + 
            ([self.synth_data_1[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        target_loader = DataLoader(self.training_graphs.shuffle() + 
            ([self.synth_data_2[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        
        return list(zip(source_loader, target_loader))

    def transform(self, data):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data["g1"] = data[0]
        new_data["g2"] = data[1]

        normalized_ged = self.nged_matrix[data[0]["i"].reshape(-1).tolist(),data[1]["i"].reshape(-1).tolist()].tolist()
        new_data["target"] = torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
        return new_data

    def process_batch(self, data):
        """
        Forward pass with a data.
        :param data: Data that is essentially pair of batches, for source and target graphs.
        :return loss: Loss on the data. 
        """
        self.optimizer.zero_grad()
        data = self.transform(data)
        target = data["target"]
        prediction = self.model(data)
        loss = F.mse_loss(prediction, target, reduction='sum')
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model.train()
        
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        loss_list = []
        loss_list_test = []
        for epoch in epochs:
            
            if self.args.plot:
                if epoch % 10 == 0:
                    self.model.train(False)
                    cnt_test = 20
                    cnt_train = 100
                    t = tqdm(total=cnt_test*cnt_train, position=2, leave=False, desc="Validation")
                    scores = torch.empty((cnt_test, cnt_train))
                    
                    for i, g in enumerate(self.testing_graphs[:cnt_test].shuffle()):
                        source_batch = Batch.from_data_list([g]*cnt_train)
                        target_batch = Batch.from_data_list(self.training_graphs[:cnt_train].shuffle())
                        data = self.transform((source_batch, target_batch))
                        target = data["target"]
                        prediction = self.model(data)
                        
                        scores[i] = F.mse_loss(prediction, target, reduction='none').detach()
                        t.update(cnt_train)
                    
                    t.close()
                    loss_list_test.append(scores.mean().item())
                    self.model.train(True)
            
            batches = self.create_batches()
            main_index = 0
            loss_sum = 0
            for index, batch_pair in tqdm(enumerate(batches), total=len(batches), desc="Batches", leave=False):
                loss_score = self.process_batch(batch_pair)
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss_score
            loss = loss_sum / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss,5))
            loss_list.append(loss)
            
        if self.args.plot:
            plt.plot(loss_list, label="Train")
            plt.plot([*range(0, self.args.epochs, 10)], loss_list_test, label="Validation")
            plt.ylim([0, 0.01])
            plt.legend()
            filename = self.args.dataset
            filename += '_' + self.args.gnn_operator 
            if self.args.diffpool:
                filename += '_diffpool'
            if self.args.histogram:
                filename += '_hist'
            filename = filename + str(self.args.epochs) + '.pdf'
            plt.savefig(filename)

    def measure_time(self):
        import time
        self.model.eval()        
        count = len(self.testing_graphs)*len(self.training_graphs)
        
        t = np.empty(count)
        i = 0
        tq = tqdm(total=count, desc="Graph pairs")
        for g1 in self.testing_graphs:
            for g2 in self.training_graphs:
                source_batch = Batch.from_data_list([g1])
                target_batch = Batch.from_data_list([g2])
                data = self.transform((source_batch, target_batch))
                
                start = time.process_time() 
                self.model(data)
                t[i] = (time.process_time() - start)
                i += 1
                tq.update()
        tq.close()
        
        print("Average time (ms): {}; Standard deviation: {}".format(round(t.mean()*1000, 5), round(t.std()*1000, 5)))
    
    def score(self):
        """
        Scoring.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()
        
        scores = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        ground_truth = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        prediction_mat = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        
        rho_list = []
        tau_list = []
        prec_at_10_list = [] 
        prec_at_20_list = []
        
        t = tqdm(total=len(self.testing_graphs)*len(self.training_graphs))

        for i, g in enumerate(self.testing_graphs):
            source_batch = Batch.from_data_list([g]*len(self.training_graphs))
            target_batch = Batch.from_data_list(self.training_graphs)
            
            data = self.transform((source_batch, target_batch))
            target = data["target"]
            ground_truth[i] = target
            prediction = self.model(data)
            prediction_mat[i] = prediction.detach().numpy()
            
            scores[i] = F.mse_loss(prediction, target, reduction='none').detach().numpy()

            rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
            tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
            prec_at_10_list.append(calculate_prec_at_k(10, prediction_mat[i], ground_truth[i]))
            prec_at_20_list.append(calculate_prec_at_k(20, prediction_mat[i], ground_truth[i]))

            t.update(len(self.training_graphs))

        self.rho = np.mean(rho_list).item()
        self.tau = np.mean(tau_list).item()
        self.prec_at_10 = np.mean(prec_at_10_list).item()
        self.prec_at_20 = np.mean(prec_at_20_list).item()
        self.model_error = np.mean(scores).item()
        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        print("\nmse(10^-3): " + str(round(self.model_error*1000, 5)) + ".")
        print("Spearman's rho: " + str(round(self.rho, 5)) + ".")
        print("Kendall's tau: " + str(round(self.tau, 5)) + ".")
        print("p@10: " + str(round(self.prec_at_10, 5)) + ".")
        print("p@20: " + str(round(self.prec_at_20, 5)) + ".")
