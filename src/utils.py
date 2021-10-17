import math
import numpy as np
import networkx as nx
import torch
import random
from texttable import Texttable
from torch_geometric.utils import erdos_renyi_graph, to_undirected, to_networkx
from torch_geometric.data import Data
import matplotlib.pyplot as plt


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows(
        [["Parameter", "Value"]]
        + [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
    )
    print(t.draw())


def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """
    temp = prediction.argsort()
    r_prediction = np.empty_like(temp)
    r_prediction[temp] = np.arange(len(prediction))

    temp = target.argsort()
    r_target = np.empty_like(temp)
    r_target[temp] = np.arange(len(target))

    return rank_corr_function(r_prediction, r_target).correlation


def calculate_prec_at_k(k, prediction, target):
    """
    Calculating precision at k.
    """

    # increase k in case same similarity score values of k-th, (k+i)-th elements
    target_increase = np.sort(target)[::-1]
    target_value_sel = (target_increase >= target_increase[k - 1]).sum()
    target_k = max(k, target_value_sel)

    best_k_pred = prediction.argsort()[::-1][:k]
    best_k_target = target.argsort()[::-1][:target_k]

    return len(set(best_k_pred).intersection(set(best_k_target))) / k


def denormalize_sim_score(g1, g2, sim_score):
    """
    Converts normalized similar into ged.
    """
    return denormalize_ged(g1, g2, -math.log(sim_score, math.e))


def denormalize_ged(g1, g2, nged):
    """
    Converts normalized ged into ged.
    """
    return round(nged * (g1.num_nodes + g2.num_nodes) / 2)


def gen_synth_data(count=200, nl=None, nu=50, p=0.5, kl=None, ku=2):
    """
    Generating synthetic data based on Erdos–Renyi model.
    :param count: Number of graph pairs to generate.
    :param nl: Minimum number of nodes in a source graph.
    :param nu: Maximum number of nodes in a source graph.
    :param p: Probability of an edge.
    :param kl: Minimum number of insert/remove edge operations on a graph.
    :param ku: Maximum number of insert/remove edge operations on a graph.
    """
    if nl is None:
        nl = nu
    if kl is None:
        kl = ku

    data = []
    data_new = []
    mat = torch.full((count, count), float("inf"))
    norm_mat = torch.full((count, count), float("inf"))

    for i in range(count):
        n = random.randint(nl, nu)
        edge_index = erdos_renyi_graph(n, p)
        x = torch.ones(n, 1)

        g1 = Data(x=x, edge_index=edge_index, i=torch.tensor([i]))
        g2, ged = gen_pair(g1, kl, ku)

        data.append(g1)
        data_new.append(g2)
        mat[i, i] = ged
        norm_mat[i, i] = ged / (0.5 * (g1.num_nodes + g2.num_nodes))

    return data, data_new, mat, norm_mat


def gen_pairs(graphs, kl=None, ku=2):
    gen_graphs_1 = []
    gen_graphs_2 = []

    count = len(graphs)
    mat = torch.full((count, count), float("inf"))
    norm_mat = torch.full((count, count), float("inf"))

    for i, g in enumerate(graphs):
        g = g.clone()
        g.i = torch.tensor([i])
        g2, ged = gen_pair(g, kl, ku)
        gen_graphs_1.append(g)
        gen_graphs_2.append(g2)
        mat[i, i] = ged
        norm_mat[i, i] = ged / (0.5 * (g.num_nodes + g2.num_nodes))

    return gen_graphs_1, gen_graphs_2, mat, norm_mat


def to_directed(edge_index):
    row, col = edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)


def gen_pair(g, kl=None, ku=2):
    if kl is None:
        kl = ku

    directed_edge_index = to_directed(g.edge_index)

    n = g.num_nodes
    num_edges = directed_edge_index.size()[1]
    to_remove = random.randint(kl, ku)

    edge_index_n = directed_edge_index[:, torch.randperm(num_edges)[to_remove:]]
    if edge_index_n.size(1) != 0:
        edge_index_n = to_undirected(edge_index_n)

    row, col = g.edge_index
    adj = torch.ones((n, n), dtype=torch.uint8)
    adj[row, col] = 0
    non_edge_index = adj.nonzero().t()

    directed_non_edge_index = to_directed(non_edge_index)
    num_edges = directed_non_edge_index.size()[1]

    to_add = random.randint(kl, ku)

    edge_index_p = directed_non_edge_index[:, torch.randperm(num_edges)[:to_add]]
    if edge_index_p.size(1):
        edge_index_p = to_undirected(edge_index_p)
    edge_index_p = torch.cat((edge_index_n, edge_index_p), 1)

    if hasattr(g, "i"):
        g2 = Data(x=g.x, edge_index=edge_index_p, i=g.i)
    else:
        g2 = Data(x=g.x, edge_index=edge_index_p)

    g2.num_nodes = g.num_nodes
    return g2, to_remove + to_add


# fmt: off
def aids_labels(g):
    types = [
        "O", "S", "C", "N", "Cl", "Br", "B", "Si", "Hg", "I", "Bi", "P", "F",
        "Cu", "Ho", "Pd", "Ru", "Pt", "Sn", "Li", "Ga", "Tb", "As", "Co", "Pb",
        "Sb", "Se", "Ni", "Te"
    ]

    return [types[i] for i in g.x.argmax(dim=1).tolist()]
# fmt: on


def draw_graphs(glist, aids=False):
    for i, g in enumerate(glist):
        plt.clf()
        G = to_networkx(g).to_undirected()
        if aids:
            label_list = aids_labels(g)
            labels = {}
            for j, node in enumerate(G.nodes()):
                labels[node] = label_list[j]
            nx.draw(G, labels=labels)
        else:
            nx.draw(G)
        plt.savefig("graph{}.png".format(i))


def draw_weighted_nodes(filename, g, model):
    """
    Draw graph with weighted nodes (for AIDS).
    """
    features = model.convolutional_pass(g.edge_index, g.x)
    coefs = model.attention.get_coefs(features)

    print(coefs)

    plt.clf()
    G = to_networkx(g).to_undirected()

    label_list = aids_labels(g)
    labels = {}
    for i, node in enumerate(G.nodes()):
        labels[node] = label_list[i]

    vmin = coefs.min().item() - 0.005
    vmax = coefs.max().item() + 0.005

    nx.draw(
        G,
        node_color=coefs.tolist(),
        cmap=plt.cm.Reds,
        labels=labels,
        vmin=vmin,
        vmax=vmax,
    )

    # sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # sm.set_array(coefs.tolist())
    # cbar = plt.colorbar(sm)

    plt.savefig(filename)
