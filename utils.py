import random
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import networkx as nx
import numpy as np
import time
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TARGET_NODE = 1000001
DIFF_COSTS = True
DIFF_ALPHA = 5.0
# DIFF_BETA = 1.0

def constructG(subsetg, preds):
    '''
    '''
    N = len(subsetg.nodes()) - 1
    M = len(subsetg.edges())
    G = to_variable(np.zeros((N,N))).float()
    Q = to_variable(np.zeros((M,N))).float()
    Gv = to_variable(np.zeros(N)).float()

    node_dict = {}
    edge_dict = {}

    nodes = list(subsetg.nodes())
    nodes.remove(TARGET_NODE)
    assert len(nodes) == N
    nodes.sort()
    for i, node in enumerate(nodes):
        node_dict[node] = i

    edges = list(subsetg.edges())
    edges.sort()
    for i, edge in enumerate(edges):
        edge_dict[edge] = i

    # node with all tables is source, node with no tables is target
    ## FIXME: need to set it appropriately.
    source_node = 0
    Gv[node_dict[source_node]] = 1.0
    # target_node = TARGET_NODE
    # Gv[node_dict[target_node]] = -1.0

    for i, node in enumerate(nodes):
        # going to set G[i,:]
        in_edges = subsetg.in_edges(node)
        out_edges = subsetg.out_edges(node)
        for edge in in_edges:
            assert edge[1] == node
            cost = preds[edge_dict[edge]]
            cost = 1.0 / cost
            cur_node_idx = node_dict[edge[1]]
            other_node_idx = node_dict[edge[0]]
            G[i,cur_node_idx] += cost
            G[i,other_node_idx] -= cost

        for edge in out_edges:
            assert edge[0] == node
            cost = preds[edge_dict[edge]]
            cost = 1.0 / cost
            cur_node_idx = node_dict[edge[0]]
            G[i,cur_node_idx] += cost

            other_node = edge[1]
            if other_node in node_dict:
                other_node_idx = node_dict[other_node]
                G[i,other_node_idx] -= cost

    for i, edge in enumerate(edges):
        cost = preds[edge_dict[edge]]
        cost = 1.0 / cost

        head_node = edge[0]
        tail_node = edge[1]
        hidx = node_dict[head_node]
        Q[i,hidx] = cost
        if tail_node in node_dict:
            tidx = node_dict[tail_node]
            Q[i,tidx] = -cost

    return edges, G, Gv, Q

def sp_loss(graph, preds, true_vals):
    edges = list(graph.edges())
    edges.sort()
    for i, e in enumerate(edges):
        graph[e[0]][e[1]]["true_cost"] = true_vals[i].item()
    opt_path = nx.shortest_path(graph, 0, TARGET_NODE, weight="true_cost")

    for i, e in enumerate(edges):
        graph[e[0]][e[1]]["est_cost"] = preds[i].item()

    est_path = nx.shortest_path(graph, 0, TARGET_NODE, weight="est_cost")
    if np.array_equal(est_path, opt_path):
        return 0.0
    else:
        return 1.0

def flow_loss(graph, preds, true_vals):
    edges,G,Gv,Q = constructG(graph, preds)
    trueC = torch.eye(len(true_vals))
    for i, y in enumerate(true_vals):
        trueC[i,i] = y
    invG = torch.inverse(G)
    left = (Gv @ torch.transpose(invG,0,1)) @ torch.transpose(Q, 0, 1)
    right = Q @ (invG @ Gv)

    print(torch.min(right))
    if torch.min(right) < 0.0:
        print(torch.min(right))
        pdb.set_trace()

    loss = left @ trueC @ right
    if loss.item() == 0.0:
        print(true_vals)
        print(preds)
        pdb.set_trace()
    # print(loss)
    # print(G.shape, Gv.shape, Q.shape)
    # print(loss)
    # pdb.set_trace()
    return loss

def eval_loss(loss_fn, net, X, Y, samples=None):
    start = time.time()
    losses = []
    l1 = []
    l2 = []
    for i, xbatch in enumerate(X):
        ybatch = Y[i]
        if net is None:
            pred = ybatch
        else:
            pred = net(xbatch).squeeze(1)
            # sample = samples[i]
            # loss = loss_fn(sample, ybatch, ybatch)
            # losses.append(loss.item())
            # continue
        # pred = net(xbatch).squeeze(1)

        if samples:
            sample = samples[i]
            loss = loss_fn(sample, pred, ybatch)
        else:
            loss = loss_fn(pred, ybatch)

        if "MSE" in str(loss_fn):
            losses_all = (pred-ybatch)**2
            losses_all = losses_all.detach().numpy()
            cur_l1 = [l for i,l in enumerate(losses_all) if i % 2 == 0]
            cur_l2 = [l for i,l in enumerate(losses_all) if i % 2 == 1]
            l1.append(np.mean(cur_l1))
            l2.append(np.mean(cur_l2))

        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        losses.append(loss)

    if "MSE" in str(loss_fn):
        print("eval loss took: ", time.time() - start)
        print("l1: {}, l2: {}, l: {}".format(np.mean(l1), np.mean(l2),
            np.mean(losses)))
        # pdb.set_trace()

    return np.mean(losses)

def get_training_features(samples, min_val, max_val, max_cost):
    train_features = []
    train_y = []
    for G in samples:
        feats, y = get_features(G, min_val, max_val, max_cost)
        train_features.append(feats)
        train_y.append(y)
    train_features = to_variable(train_features).float()
    train_y= to_variable(train_y).float()
    return train_features, train_y

def get_features(G, min_val, max_val, max_cost):
    '''
    feature vector for each edge in G.
    '''
    feats = []
    y = []
    edges = list(G.edges())
    edges.sort()
    for edge in edges:
        n1_val = G.nodes()[edge[0]]["val"]
        n2_val = G.nodes()[edge[1]]["val"]
        cost = G[edge[0]][edge[1]]["cost"]
        if min_val is not None:
            n1_val = (n1_val - min_val) / (max_val - min_val)
            n2_val = (n2_val - min_val) / (max_val - min_val)
            cost = cost / max_cost
        # assert cost >= 0.0 and cost <= 1.00
        feats.append([n1_val, n2_val])
        y.append(cost)
    return feats, y

def update_all_nodes(G, min_val, max_val):
    target = TARGET_NODE
    G.add_node(target)

    # add a value to each source node
    random.seed(1234)
    for node in G.nodes():
        G.nodes()[node]["val"] = float(random.randint(min_val, max_val))
        if G.out_degree(node) == 0 and node != target:
            G.add_edge(node, target)

def compute_costs(G, DIFF_COSTS=True):
    '''
    '''
    for i, edge in enumerate(G.edges()):
        n1_val = G.nodes()[edge[0]]["val"]
        n2_val = G.nodes()[edge[1]]["val"]

        if DIFF_COSTS:
            if i % 2 == 0:
                G[edge[0]][edge[1]]["cost"] = n1_val + n2_val
                # G[edge[0]][edge[1]]["cost"] = n1_val*n2_val
            elif i % 2 == 1:
                G[edge[0]][edge[1]]["cost"] = DIFF_ALPHA*n1_val

            # if i % 3 == 0:
                # G[edge[0]][edge[1]]["cost"] = n1_val + n2_val
            # elif i % 3 == 1:
                # G[edge[0]][edge[1]]["cost"] = DIFF_ALPHA*n1_val
            # else:
                # G[edge[0]][edge[1]]["cost"] = DIFF_BETA*n2_val
        else:
            G[edge[0]][edge[1]]["cost"] = n1_val + n2_val
            # G[edge[0]][edge[1]]["cost"] = n1_val*n2_val

def gen_dataset(N, width, height, min_val=1, max_val=100):
    samples = []
    for i in range(N):
        if i % 100 == 0:
            print("sample: ", i)
        G = nx.balanced_tree(width,height,create_using=nx.DiGraph())
        update_all_nodes(G, min_val=min_val, max_val=max_val)
        compute_costs(G, DIFF_COSTS=DIFF_COSTS)
        samples.append(G)
    return samples

class SimpleRegression(torch.nn.Module):
    # TODO: add more stuff?
    def __init__(self, input_width, hidden_width_multiple,
            n_output, num_hidden_layers=1, hidden_layer_size=None):
        super(SimpleRegression, self).__init__()
        if hidden_layer_size is None:
            n_hidden = int(input_width * hidden_width_multiple)
        else:
            n_hidden = hidden_layer_size

        self.layers = []
        self.layer1 = nn.Sequential(
            nn.Linear(input_width, n_hidden, bias=True),
            nn.LeakyReLU()
        ).to(device)
        self.layers.append(self.layer1)

        for i in range(0,num_hidden_layers-1,1):
            layer = nn.Sequential(
                nn.Linear(n_hidden, n_hidden, bias=True),
                nn.LeakyReLU()
            ).to(device)
            self.layers.append(layer)

        self.final_layer = nn.Sequential(
            nn.Linear(n_hidden, n_output, bias=True),
            nn.Sigmoid()
        ).to(device)
        self.layers.append(self.final_layer)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

def to_variable(arr, use_cuda=True, requires_grad=False):
    if isinstance(arr, list) or isinstance(arr, tuple):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        arr = Variable(torch.from_numpy(arr), requires_grad=requires_grad).to(device)
    else:
        arr = Variable(arr, requires_grad=requires_grad).to(device)

    # if torch.cuda.is_available() and use_cuda:
        # print("returning cuda array!")
        # arr = arr.cuda()
    # else:
        # pdb.set_trace()
    return arr
