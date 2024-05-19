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
        '''print("### 333 after rgcn 1st conv layer ###")
        for column in node_types:
            print(h[column].shape)'''
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
        self.mlp = MLP(hidden_dim, hidden_dim, n_classes)

    def forward(self, graphs):
        all_graph_pooling_feat = torch.zeros(len(graphs), hidden_dim).to(device)
        for index, g in enumerate(graphs):
            h = g.ndata['feat']
            '''print("### 111 raw features ###")
            for column in node_types:
                print(h[column].shape)'''
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
node_types = ['Src_Type', 'Dst_Type']

graph_rel_names, _ = dgl.load_graphs('../data/streamspot/single_graph/rel_names.dgl')
graph_rel_names = graph_rel_names[0]
rel_names = graph_rel_names.etypes

# 划分训练集、测试集、验证集
def divide_train_val_test():
    # 图300-399为正例，其余为负例
    all_pos = np.arange(300, 400)
    part_1_neg = np.arange(0, 300)
    part_2_neg = np.arange(400, 600)
    all_neg = np.append(part_1_neg, part_2_neg)

    test_pos_idx = np.random.choice(all_pos, 40, replace=False)
    test_neg_idx = np.random.choice(all_neg, 200, replace=False)
    test_idx = np.append(test_pos_idx, test_neg_idx)

    train_val_pos_idx = all_pos[~np.isin(all_pos, test_pos_idx)]
    train_val_neg_idx = all_neg[~np.isin(all_neg, test_neg_idx)]

    train_pos_idx = np.random.choice(train_val_pos_idx, 40, replace=False)
    train_neg_idx = np.random.choice(train_val_neg_idx, 200, replace=False)
    train_idx = np.append(train_pos_idx, train_neg_idx)

    val_pos_idx = train_val_pos_idx[~np.isin(train_val_pos_idx, train_pos_idx)]
    val_neg_idx = train_val_neg_idx[~np.isin(train_val_neg_idx, train_neg_idx)]
    val_idx = np.append(val_pos_idx, val_neg_idx)

    return train_idx, val_idx, test_idx


def streamspot_train_and_eval(num_epoch):
    learning_rate = 0.001
    train_idx, val_idx, test_idx = divide_train_val_test()
    # 加载训练集
    train_graphs = []
    for i in train_idx:
        loaded_graph, _ = dgl.load_graphs('../data/streamspot/single_graph/streamspot_' + str(i) +'.dgl')
        loaded_graph = loaded_graph[0].to(device)
        train_graphs.append(loaded_graph)

    train_y = np.where(train_idx >= 300, train_idx, 600)
    train_y = np.where(train_y <= 399, 1, 0)
    train_y = torch.tensor(train_y).long().to(device)
    # 加载验证集
    val_graphs = []
    for i in val_idx:
        loaded_graph, _ = dgl.load_graphs('../data/streamspot/single_graph/streamspot_' + str(i) +'.dgl')
        loaded_graph = loaded_graph[0].to(device)
        val_graphs.append(loaded_graph)
    val_y = np.where(val_idx >= 300, val_idx, 600)
    val_y = np.where(val_y <= 399, 1, 0)
    val_y = torch.tensor(val_y).long().to(device)
    # 训练模型
    model = Aggregationfeature(raw_feat_dim, hidden_dim, 2, rel_names).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_acc = 0.0
    for epoch in range(num_epoch):
        model.train()
        train_y_hat = model(train_graphs)
        loss = F.cross_entropy(train_y_hat, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        acc = evaluate(val_graphs, val_y, model)
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), 'strcture_best_model.pt')
            np.save('../data/streamspot/best_test_idx', test_idx)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )
    # 加载测试数据集
    test_graphs = []
    for i in test_idx:
        loaded_graph, _ = dgl.load_graphs('../data/streamspot/single_graph/streamspot_' + str(i) +'.dgl')
        loaded_graph = loaded_graph[0].to(device)
        test_graphs.append(loaded_graph)
    test_y = np.where(test_idx >= 300, test_idx, 600)
    test_y = np.where(test_y <= 399, 1, 0)
    test_y_list = []
    test_y_hat_list = []
    # 评估模型
    report = ""
    model.load_state_dict(torch.load('strcture_best_model.pt'))
    model.eval()
    with torch.no_grad():
        test_y_hat = model(test_graphs)
        test_y_hat = torch.argmax(test_y_hat, dim = 1)
        test_y_list.extend(test_y.tolist())
        test_y_hat_list.extend(test_y_hat.tolist())
        report = classification_report(test_y_list, test_y_hat_list,  digits=5)
        with open("../report/streamspot_report.txt", "a") as report_file:
            report_file.write(report)
    report_array = str.split(report)
    accuracy = float(report_array[25])
    recall = float(report_array[11])
    return accuracy, recall


def streamspot_eval():
    test_idx = np.load('../data/streamspot/best_test_idx.npy')
    test_graphs = []
    for i in test_idx:
        loaded_graph, _ = dgl.load_graphs('../data/streamspot/single_graph/streamspot_' + str(i) +'.dgl')
        loaded_graph = loaded_graph[0].to(device)
        test_graphs.append(loaded_graph)
    test_y = np.where(test_idx >= 300, test_idx, 600)
    test_y = np.where(test_y <= 399, 1, 0)

    test_y_list = []
    test_y_hat_list = []

    model = Aggregationfeature(raw_feat_dim, hidden_dim, 2, rel_names).to(device)
    model.load_state_dict(torch.load('strcture_best_model.pt'))
    model.eval()
    with torch.no_grad():
        test_y_hat = model(test_graphs)
        test_y_hat = torch.argmax(test_y_hat, dim = 1)
        test_y_list.extend(test_y.tolist())
        test_y_hat_list.extend(test_y_hat.tolist())
        report = classification_report(test_y_list, test_y_hat_list,  digits=5)
        print("Classification Report:\n", report)


if __name__ == "__main__":
    '''num_train_and_eval = 10
    num_epoch = 30
    accuracy_accumulator = []
    recall_accumulator = []
    
    for i in range(num_train_and_eval):
        accuracy, recall = streamspot_train_and_eval(num_epoch)
        accuracy_accumulator.append(accuracy) 
        recall_accumulator.append(recall)

    print("{num_train_and_eval} times train and evaluation report:")
    print("time\t\taccuracy\t\trecall")
    for i in range(num_train_and_eval):
        print(i, "\t\t", accuracy_accumulator[i], "\t\t", recall_accumulator[i])
    print("avg\t\t", np.mean(accuracy_accumulator), "\t\t", np.mean(recall_accumulator))'''
    streamspot_eval()