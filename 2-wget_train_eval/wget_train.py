import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

# 关系图卷积神经网络
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

# 多层感知机
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

# 整体模型
class Aggregationfeature(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()
        # Linear layer to compress input features from in_dim to hidden_dim
        self.input_linear = nn.Linear(in_dim, hidden_dim)
        self.rgcn = RGCN(hidden_dim, hidden_dim, hidden_dim, rel_names)
        self.mlp = MLP(hidden_dim, 64, n_classes)

    def forward(self, graphs):
        all_graph_pooling_feat = torch.zeros(len(graphs), hidden_dim).to(device)
        for index, g in enumerate(graphs):
            h = g.ndata['feat']
            h = {k: F.relu(self.input_linear(v.to(device))) for k, v in h.items()}
            h = self.rgcn(g, h)
            with g.local_scope():
                g.ndata['h'] = h
                g_sum_feat = torch.zeros(hidden_dim).to(device)
                num = 0
                for _, node_type_tensor in h.items():
                    for attribute_value in node_type_tensor:
                        num += 1
                        g_sum_feat += attribute_value
                graph_pooling_feat = g_sum_feat / num
                all_graph_pooling_feat[index, :] = graph_pooling_feat
                # print("### graph ", index, " ###", graph_pooling_feat)
        return self.mlp(all_graph_pooling_feat)

# 评估函数
def evaluate(graphs, val_y, model):
    model.eval()
    with torch.no_grad():
        val_y_hat = model(graphs)
        _, indices = torch.max(val_y_hat, dim=1)
        correct = torch.sum(indices == val_y)
        return correct.item() * 1.0 / len(val_y)

device = torch.device('cuda:0')
raw_feat_dim = 768
hidden_dim = 128

graph_rel_names, _ = dgl.load_graphs('../data/wget/wget_graph/rel_names.dgl')
graph_rel_names = graph_rel_names[0]
rel_names = graph_rel_names.etypes


def wget_train(train_pos, train_neg, test_pos, test_neg, num_epoch, batch_size):
    learning_rate = 0.001

    # 训练模型
    model = Aggregationfeature(raw_feat_dim, hidden_dim, 2, rel_names).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        model.train()
        for i in range(batch_size):
            model = wget_batch_train(model, opt, train_pos, train_neg, epoch, i)
    
    torch.save(model.state_dict(), 'strcture_model.pt')
    np.save('../data/wget/test_idx', np.append(test_pos, test_neg))
    accuracy, recall = wget_eval(test_pos, test_neg)
    return accuracy, recall


def wget_batch_train(model, opt, train_pos, train_neg, epoch, ith_batch):
    # 加载训练集
    train_graphs = []
    for i in train_pos[ith_batch * 5: (ith_batch + 1) * 5]:
        loaded_graph, _ = dgl.load_graphs('../data/wget/wget_graph/wget-baseline-attack-' + str(i) +'.dgl')
        loaded_graph = loaded_graph[0].to(device)
        train_graphs.append(loaded_graph)
    for i in train_neg[ith_batch * 25: (ith_batch + 1) * 25]:
        loaded_graph, _ = dgl.load_graphs('../data/wget/wget_graph/wget-normal-' + str(i) +'.dgl')
        loaded_graph = loaded_graph[0].to(device)
        train_graphs.append(loaded_graph)
    train_y = np.append(np.ones(5), np.zeros(25))
    train_y = torch.tensor(train_y).long().to(device)

    model.train()
    train_y_hat = model(train_graphs)
    loss = F.cross_entropy(train_y_hat, train_y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    '''train_y_list = []
    train_y_hat_list = []
    # 训练集评估
    model.eval()
    with torch.no_grad():
        train_y_hat = model(train_graphs)
        train_y_hat = torch.argmax(train_y_hat, dim = 1)
        train_y_list.extend(train_y.tolist())
        train_y_hat_list.extend(train_y_hat.tolist())
        report = classification_report(train_y_list, train_y_hat_list,  digits = 5)
    report_array = str.split(report)
    accuracy = float(report_array[25])
    recall = float(report_array[11])'''
    print(
        "Epoch {:05d} | Batch {:03d} | Loss {:.4f}".format(
            epoch, ith_batch, loss.item()
        )
    )
    return model


def wget_eval(test_pos, test_neg):
    model = Aggregationfeature(raw_feat_dim, hidden_dim, 2, rel_names).to(device)
    # 加载测试数据集
    test_graphs = []
    for i in test_pos:
        loaded_graph, _ = dgl.load_graphs('../data/wget/wget_graph/wget-baseline-attack-' + str(i) +'.dgl')
        loaded_graph = loaded_graph[0].to(device)
        test_graphs.append(loaded_graph)
    for i in test_neg:
        loaded_graph, _ = dgl.load_graphs('../data/wget/wget_graph/wget-normal-' + str(i) +'.dgl')
        loaded_graph = loaded_graph[0].to(device)
        test_graphs.append(loaded_graph)
    test_y = np.append(np.ones(5), np.zeros(25))
    test_y_list = []
    test_y_hat_list = []

    # 评估模型
    report = ""
    model.load_state_dict(torch.load('strcture_model.pt'))
    model.eval()
    with torch.no_grad():
        test_y_hat = model(test_graphs)
        test_y_hat = torch.argmax(test_y_hat, dim = 1)
        test_y_list.extend(test_y.tolist())
        test_y_hat_list.extend(test_y_hat.tolist())
        report = classification_report(test_y_list, test_y_hat_list,  digits = 5)
        with open("../report/wget_report.txt", "a") as report_file:
            report_file.write(report)
    report_array = str.split(report)
    accuracy = float(report_array[25])
    recall = float(report_array[11])
    return accuracy, recall


def wget_evaluate():
    test_idx = np.load('../data/wget/test_idx.npy')
    test_graphs = []
    for i in test_idx[0: 5]:
        loaded_graph, _ = dgl.load_graphs('../data/wget/wget_graph/wget-baseline-attack-' + str(i) +'.dgl')
        loaded_graph = loaded_graph[0].to(device)
        test_graphs.append(loaded_graph)
    for i in test_idx[5: 30]:
        loaded_graph, _ = dgl.load_graphs('../data/wget/wget_graph/wget-normal-' + str(i) +'.dgl')
        loaded_graph = loaded_graph[0].to(device)
        test_graphs.append(loaded_graph)
    test_y = np.append(np.ones(5), np.zeros(25))
    test_y_list = []
    test_y_hat_list = []

    model = Aggregationfeature(raw_feat_dim, hidden_dim, 2, rel_names).to(device)
    model.load_state_dict(torch.load('strcture_model.pt'))
    model.eval()
    with torch.no_grad():
        test_y_hat = model(test_graphs)
        test_y_hat = torch.argmax(test_y_hat, dim = 1)
        test_y_list.extend(test_y.tolist())
        test_y_hat_list.extend(test_y_hat.tolist())
        report = classification_report(test_y_list, test_y_hat_list,  digits=5)
        print("Classification Report:\n", report)


if __name__ == "__main__":
    '''num_train_and_eval = 5
    num_epoch = 50
    batch_size = 4
    accuracy_accumulator = []
    recall_accumulator = []
    
    # 划分训练集和测试集
    all_pos = np.arange(0, 25)
    all_neg = np.arange(0, 125)
    np.random.shuffle(all_pos)
    np.random.shuffle(all_neg)

    # 五折交叉验证
    for i in range(num_train_and_eval):
        test_pos = all_pos[i * 5: (i + 1) * 5]
        test_neg = all_neg[i * 25: (i + 1) * 25]
        train_pos = np.append(all_pos[0: i * 5], all_pos[(i + 1) * 5: ])
        train_neg = np.append(all_neg[0: i * 25], all_neg[(i + 1) * 25: ])
        accuracy, recall = wget_train(train_pos, train_neg, test_pos, test_neg, num_epoch, batch_size)
        accuracy_accumulator.append(accuracy)
        recall_accumulator.append(recall)

    print("{num_train_and_eval} times train and evaluation report:")
    print("time\t\taccuracy\t\trecall")
    for i in range(num_train_and_eval):
        print(i, "\t\t", accuracy_accumulator[i], "\t\t", recall_accumulator[i])
    print("avg\t\t", np.mean(accuracy_accumulator), "\t\t", np.mean(recall_accumulator))'''
    wget_evaluate()
