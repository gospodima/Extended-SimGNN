import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau

from .layers import AttentionModule, TensorNetworkModule, DiffPool
from .utils import calculate_ranking_correlation, calculate_prec_at_k, gen_pairs

from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
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
        self.args = args    #命令行所输入的所有参数
        self.number_labels = number_of_labels   #点的标签数 =29
        self.setup_layers()

    def calculate_bottleneck_features(self):    #计算瓶颈特征的形状
        """
        Deciding the shape of the bottleneck layer.
        在深度学习中，瓶颈特征（bottleneck features）通常指的是在卷积神经网络（CNN）中，位于网络的中间层的特征表示。这些特征表示在网络的前向传播过程中，经过了一系列卷积和池化等操作，并在接近输出层之前的某一层进行了压缩和抽象。

        计算瓶颈特征的目的是提取出图像或输入数据的高级表示，这些表示往往更具有判别性和泛化能力。瓶颈特征可以作为输入数据的一种编码形式，用于训练分类器、回归模型或其他机器学习模型。
        """
        if self.args.histogram: #histogram：是否包括直方图特征，默认为false
            self.feature_count = self.args.tensor_neurons + self.args.bins  #非直方图特征的数量（NTN的数量）+直方图特征的数量
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        if self.args.gnn_operator == "gcn": # GCN（Graph Convolutional Network）图卷积网络
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1) #标签数[29,64]
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)#[64,32]
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)#[32,16]
        elif self.args.gnn_operator == "gin":   # GIN（Graph Isomorphism Network）图同构网络
            # 使用多个线性层和批标准化层构建了三个序列化的神经网络 nn1、nn2 和 nn3。使用 GINConv 类创建三个 GIN 层，这些层的输入和输出网络由相应的 nn 对象和 train_eps=True 参数指定。
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1),   #[29,64]
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),  #[64,64]
                torch.nn.BatchNorm1d(self.args.filters_1),
            )

            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2),  #[64,32]
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),  #[32,32]
                torch.nn.BatchNorm1d(self.args.filters_2),
            )

            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3),  #[32,16]
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),  #[16,16]
                torch.nn.BatchNorm1d(self.args.filters_3),
            )

            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        else:
            raise NotImplementedError("Unknown GNN-Operator.")

        if self.args.diffpool:  #选择不同的池化层（Pooling Layer），默认false
            self.attention = DiffPool(self.args)
        else:
            self.attention = AttentionModule(self.args)

        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(   #创建一个全连接层 fully_connected_first，它将瓶颈特征及其大小作为输入特征，输出特征的维度由参数 self.args.bottle_neck_neurons 指定。
            self.feature_count, self.args.bottle_neck_neurons
        )
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)  #创建一个线性层 scoring_layer，将全连接层的输出作为输入特征，输出特征维度为 1。

    def calculate_histogram(
        self, abstract_features_1, abstract_features_2, batch_1, batch_2
    ):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for target graphs.
        :param abstract_features_2: Feature matrix for source graphs.
        :param batch_1: Batch vector for source graphs, which assigns each node to a specific example
        :param batch_1: Batch vector for target graphs, which assigns each node to a specific example
        :return hist: Histsogram of similarity scores.
        """
        abstract_features_1, mask_1 = to_dense_batch(abstract_features_1, batch_1)  #to_dense_batch中，batch表示每个样本所属的批次
        abstract_features_2, mask_2 = to_dense_batch(abstract_features_2, batch_2)

        B1, N1, _ = abstract_features_1.size()  # [128,10,16]
        B2, N2, _ = abstract_features_2.size()  # [128,10,16]

        mask_1 = mask_1.view(B1, N1)    # view函数在pytorch中等价于reshape函数
        mask_2 = mask_2.view(B2, N2)
        num_nodes = torch.max(mask_1.sum(dim=1), mask_2.sum(dim=1)) # 一对graph pair中最大节点数组成的tensor

        scores = torch.matmul(
            abstract_features_1, abstract_features_2.permute([0, 2, 1]) # 内积计算相似度得分 [128,10,10]
        ).detach()

        hist_list = []
        for i, mat in enumerate(scores):
            mat = torch.sigmoid(mat[: num_nodes[i], : num_nodes[i]]).view(-1)
            hist = torch.histc(mat, bins=self.args.bins)
            hist = hist / torch.sum(hist)
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
        batch_1 = (
            data["g1"].batch
            if hasattr(data["g1"], "batch")     # 检查 data["g1"] 对象是否具有名为 "batch" 的属性
            else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
        )
        batch_2 = (
            data["g2"].batch
            if hasattr(data["g2"], "batch")
            else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)
        )

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        if self.args.histogram:
            hist = self.calculate_histogram(
                abstract_features_1, abstract_features_2, batch_1, batch_2
            )

        if self.args.diffpool:
            pooled_features_1 = self.diffpool(
                abstract_features_1, edge_index_1, batch_1
            )
            pooled_features_2 = self.diffpool(
                abstract_features_2, edge_index_2, batch_2
            )
        else:
            pooled_features_1 = self.attention(abstract_features_1, batch_1)
            pooled_features_2 = self.attention(abstract_features_2, batch_2)

        scores = self.tensor_network(pooled_features_1, pooled_features_2)

        if self.args.histogram:
            scores = torch.cat((scores, hist), dim=1)

        scores = F.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores)).view(-1)  # [128,]
        return score


class SimGNNTrainer(object):
    """
    SimGNN model trainer.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args    # args是使用argparse模块解析命令行参数后生成的 Namespace 对象。Namespace 是 argparse 模块中的一个类，它用于存储解析后的命令行参数的值。
        self.process_dataset()  #下载并处理数据集
        self.setup_model()  #启动SimGNN模型

    def setup_model(self):
        """
        Creating a SimGNN.
        """
        self.model = SimGNN(self.args, self.number_of_labels)   #从命令行中获取的参数，标签数目

    def save(self):
        """
        Saving model.
        """
        torch.save(self.model.state_dict(), self.args.save)
        print(f"Model is saved under {self.args.save}.")

    def load(self):
        """
        Loading model.
        """
        self.model.load_state_dict(torch.load(self.args.load))
        print(f"Model is loaded from {self.args.save}.")

    def process_dataset(self):
        """
        Downloading and processing dataset.
        700个分子图，每个图的节点数不大于10，每个节点的的特征向量长度为29（29中类型的节点，独热编码）
        """
        print("\nPreparing dataset.\n")

        self.training_graphs = GEDDataset(
            "datasets/{}".format(self.args.dataset), self.args.dataset, train=True
        )
        self.testing_graphs = GEDDataset(
            "datasets/{}".format(self.args.dataset), self.args.dataset, train=False
        )
        self.nged_matrix = self.training_graphs.norm_ged    #将训练数据的规范化的图编辑距离矩阵赋值给nged_matrix =[700*700]，表示图和图之间的编辑距离矩阵(700个图）
        self.real_data_size = self.nged_matrix.size(0)  #真实数据集的大小 =700

        if self.args.synth:     #是否需要生成合成数据 =False
            # self.synth_data_1, self.synth_data_2, _, synth_nged_matrix = gen_synth_data(500, 10, 12, 0.5, 0, 3)
            self.synth_data_1, self.synth_data_2, _, synth_nged_matrix = gen_pairs(
                self.training_graphs.shuffle()[:500], 0, 3
            )   #使用 gen_pairs 函数生成一对合成数据，并将结果分别存储在 self.synth_data_1 和 self.synth_data_2 中，合成数据的规范化图编辑距离矩阵存储在 synth_nged_matrix 中。

            real_data_size = self.nged_matrix.size(0)
            synth_data_size = synth_nged_matrix.size(0)
            self.nged_matrix = torch.cat(
                (
                    self.nged_matrix,
                    torch.full((real_data_size, synth_data_size), float("inf")),
                ),
                dim=1,
            )   #在 self.nged_matrix 的列维度上拼接了一个值为正无穷的矩阵，用于将真实数据集与合成数据集的图编辑距离矩阵进行扩展
            synth_nged_matrix = torch.cat(
                (
                    torch.full((synth_data_size, real_data_size), float("inf")),
                    synth_nged_matrix,
                ),
                dim=1,
            )   #在 synth_nged_matrix 的列维度上拼接了一个值为正无穷的矩阵，用于将合成数据集与真实数据集的图编辑距离矩阵进行扩展。
            self.nged_matrix = torch.cat((self.nged_matrix, synth_nged_matrix)) #将扩展后的合成数据集和真实数据集的图编辑距离矩阵进行拼接，得到最终的图编辑距离矩阵

        if self.training_graphs[0].x is None:   #检查第一个图对象的x属性是否为 None。如果为 None，表示图对象中没有节点特征。 =[10,29]（每个图有<=10个节点，每个节点被标记为29种类型中的一种）
            max_degree = 0
            for g in (
                self.training_graphs
                + self.testing_graphs
                + (self.synth_data_1 + self.synth_data_2 if self.args.synth else [])
            ):
                if g.edge_index.size(1) > 0:
                    max_degree = max(
                        max_degree, int(degree(g.edge_index[0]).max().item())
                    )
            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.training_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree
            #计算了图对象中的最大度数，并使用 OneHotDegree 对象创建了一个独热编码的节点度数表示。然后，将这个度数表示赋值给训练数据集和测试数据集的 transform 属性

            # labeling of synth data according to real data format
            if self.args.synth: #若为True，则将合成数据集中的图对象的索引 i 增加真实数据集大小的值，用于标识合成数据在整个数据集中的位置。
                for g in self.synth_data_1 + self.synth_data_2:
                    g = one_hot_degree(g)
                    g.i = g.i + real_data_size
        elif self.args.synth:   #若为True，则将合成数据集中的图对象的索引 i 增加真实数据集大小的值，并将节点特征 x 进行扩展。
            for g in self.synth_data_1 + self.synth_data_2:
                g.i = g.i + real_data_size
                # g.x = torch.cat((g.x, torch.zeros((g.x.size(0), self.training_graphs.num_features-1))), dim=1)

        self.number_of_labels = self.training_graphs.num_features   #将训练数据集的节点特征数量作为标签的数量进行保存 =29

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: Zipped loaders as list.
        """
        if self.args.synth:
            synth_data_ind = random.sample(range(len(self.synth_data_1)), 100)

        source_loader = DataLoader(
            self.training_graphs.shuffle()
            + (
                [self.synth_data_1[i] for i in synth_data_ind]
                if self.args.synth
                else []
            ),
            batch_size=self.args.batch_size,
        )
        target_loader = DataLoader(
            self.training_graphs.shuffle()
            + (
                [self.synth_data_2[i] for i in synth_data_ind]
                if self.args.synth
                else []
            ),
            batch_size=self.args.batch_size,
        )

        return list(zip(source_loader, target_loader))

    def transform(self, data):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.通常表示为一个包含两个图的元组（tuple），每个图通常是一个字典（dictionary），包含以下关键字：
            "i"：节点索引，表示图中每个节点的唯一标识。
            "x"：节点特征，表示图中每个节点的特征向量。
            "edge_index"：边索引，表示图中边的连接关系
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data["g1"] = data[0]
        new_data["g2"] = data[1]

        normalized_ged = self.nged_matrix[  #根据图对中的节点索引（data[0]["i"] 和 data[1]["i"]），从 self.nged_matrix 中获取相应的归一化 GED（Graph Edit Distance）值，并将其转换为 Python 列表（list）。
            data[0]["i"].reshape(-1).tolist(), data[1]["i"].reshape(-1).tolist()
        ].tolist()
        new_data["target"] = (  #将归一化 GED 值取指数（exp）的负值，并将其转换为 PyTorch 张量（tensor），然后设置为字典中的键 "target"。
            torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
        )
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
        prediction = self.model(data)   #这里调用SimGNN类的forward函数
        loss = F.mse_loss(prediction, target, reduction="sum")
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.model.train()

        epochs = trange(self.args.epochs, leave=True, desc="Epoch") #创建一个可迭代的对象 epochs
        # self.args.epochs：迭代总次数，leave：在迭代完成后是否保留进度条，desc：进度条的描述文本
        loss_list = []
        loss_list_test = []
        for epoch in epochs:

            if self.args.plot:
                if epoch % 10 == 0:
                    self.model.train(False)
                    cnt_test = 20
                    cnt_train = 100
                    t = tqdm(
                        total=cnt_test * cnt_train,
                        position=2,
                        leave=False,
                        desc="Validation",
                    )
                    scores = torch.empty((cnt_test, cnt_train))

                    for i, g in enumerate(self.testing_graphs[:cnt_test].shuffle()):
                        source_batch = Batch.from_data_list([g] * cnt_train)
                        target_batch = Batch.from_data_list(
                            self.training_graphs[:cnt_train].shuffle()
                        )
                        data = self.transform((source_batch, target_batch))
                        target = data["target"]
                        prediction = self.model(data)

                        scores[i] = F.mse_loss(
                            prediction, target, reduction="none"
                        ).detach()
                        t.update(cnt_train)

                    t.close()
                    loss_list_test.append(scores.mean().item())
                    self.model.train(True)

            batches = self.create_batches() # 创建批次数据
            main_index = 0  #初始化主索引
            loss_sum = 0
            for index, batch_pair in tqdm(  #过迭代器 batches 遍历每个批次数据，并使用 tqdm 创建一个进度条
                enumerate(batches), total=len(batches), desc="Batches", leave=False
            ):
                loss_score = self.process_batch(batch_pair) # 调用 process_batch 方法对批次数据进行处理
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss_score
            loss = loss_sum / main_index    #计算平均损失
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
            loss_list.append(loss)

        if self.args.plot:
            plt.plot(loss_list, label="Train")
            plt.plot(
                [*range(0, self.args.epochs, 10)], loss_list_test, label="Validation"
            )
            plt.ylim([0, 0.01])
            plt.legend()
            filename = self.args.dataset
            filename += "_" + self.args.gnn_operator
            if self.args.diffpool:
                filename += "_diffpool"
            if self.args.histogram:
                filename += "_hist"
            filename = filename + str(self.args.epochs) + ".pdf"
            plt.savefig(filename)

    def measure_time(self):
        import time

        self.model.eval()
        count = len(self.testing_graphs) * len(self.training_graphs)

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
                t[i] = time.process_time() - start
                i += 1
                tq.update()
        tq.close()

        print(
            "Average time (ms): {}; Standard deviation: {}".format(
                round(t.mean() * 1000, 5), round(t.std() * 1000, 5)
            )
        )

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

        t = tqdm(total=len(self.testing_graphs) * len(self.training_graphs))

        for i, g in enumerate(self.testing_graphs):
            source_batch = Batch.from_data_list([g] * len(self.training_graphs))
            target_batch = Batch.from_data_list(self.training_graphs)

            data = self.transform((source_batch, target_batch))
            target = data["target"]
            ground_truth[i] = target
            prediction = self.model(data)
            prediction_mat[i] = prediction.detach().numpy()

            scores[i] = (
                F.mse_loss(prediction, target, reduction="none").detach().numpy()
            )

            rho_list.append(
                calculate_ranking_correlation(
                    spearmanr, prediction_mat[i], ground_truth[i]
                )
            )
            tau_list.append(
                calculate_ranking_correlation(
                    kendalltau, prediction_mat[i], ground_truth[i]
                )
            )
            prec_at_10_list.append(
                calculate_prec_at_k(10, prediction_mat[i], ground_truth[i])
            )
            prec_at_20_list.append(
                calculate_prec_at_k(20, prediction_mat[i], ground_truth[i])
            )

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
        print("\nmse(10^-3): " + str(round(self.model_error * 1000, 5)) + ".")
        print("Spearman's rho: " + str(round(self.rho, 5)) + ".")
        print("Kendall's tau: " + str(round(self.tau, 5)) + ".")
        print("p@10: " + str(round(self.prec_at_10, 5)) + ".")
        print("p@20: " + str(round(self.prec_at_20, 5)) + ".")
