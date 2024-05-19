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
import os

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
        self.mlp = MLP(hidden_dim, hidden_dim, n_classes)

    def forward(self, g, data):
        h = g.ndata['feat']
        '''for i, j in h.items():
            print(i, " ", j.size())'''
        h = {k: F.relu(self.input_linear(v)) for k, v in h.items()}
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            all_mean_feat = torch.zeros(len(data), hidden_dim).to(device)
            ###
            '''for feature in features:
                if feature in h:
                    print(h[feature])'''
            ###
            for index, row in data.iterrows():
                feat_sum = torch.zeros(hidden_dim).to(device)
                num = 0
                for feature in features:
                    if pd.notna(row[feature]):
                        if feature in feat_dict:
                            feat_sum += h[feature][feat_dict[feature][str(row[feature])]]
                            num = num + 1
                        '''if index == 0:
                            print("h[", feature, "][feat_dict[", feature, "][", str(row[feature]), "]] = ", h[feature][feat_dict[feature][str(row[feature])]])'''
                # 每个行为表示为属性表征的均值
                mean_feat = feat_sum / num
                all_mean_feat[index, :] = mean_feat
            return self.mlp(all_mean_feat)

# 评估函数
def evaluate(g,  labels, mask, model, sub_data):
    model.eval()
    with torch.no_grad():
        logits = model(g, sub_data)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

device = torch.device('cuda:2')
raw_feat_dim = 768
hidden_dim = 128

data_1 = pd.read_csv("../data/darpa/darpa_csv/trace-benign.csv")
data_2 = pd.read_csv("../data/darpa/darpa_csv/trace-malicious.csv")
sub_data_1 = data_1.sample(n = 25000, random_state = 1).reset_index(drop = True)
sub_data_2 = data_2.sample(n = 25000, random_state = 1).reset_index(drop = True)
data_list = [sub_data_1, sub_data_2]
data = pd.concat(data_list, axis = 0, ignore_index = True)
# data = pd.read_csv("../data/darpa/darpa_csv/ta1-trace-e3-official-1.csv")

features = [
    "src_type", "src_process_name", "src_file_path", "src_remote_address", "src_memory_size", 
    "edge_type", 
    "dst_type", "dst_process_name", "dst_file_path", "dst_remote_address", "dst_memory_size"
]
label = 'label'

with open("../data/darpa/darpa_column_encoding/ta1-trace-e3-official-1.csv.json") as file:
    feat_dict = json.load(file)

def darpa_train(i, g, rel_names, num_epoch, learning_rate):
    sub_data = data
    # sub_data = data.sample(n = 50000, random_state = i).reset_index(drop = True)
    X = data[features]
    y = torch.tensor(sub_data["label"].values, dtype = torch.long).to(device)
    train_val_test_idx = np.arange(len(sub_data))
    # 训练集60% 验证集30% 测试集20%
    train_idx, test_idx = train_test_split(train_val_test_idx, test_size = 0.2, random_state = i)
    train_idx, val_idx = train_test_split(train_idx, test_size = 0.2, random_state = i)
    train_mask = torch.zeros(len(sub_data), dtype = torch.bool)
    val_mask = torch.zeros(len(sub_data), dtype = torch.bool)
    test_mask = torch.zeros(len(sub_data), dtype = torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
        
    model = Aggregationfeature(raw_feat_dim, hidden_dim, 2, rel_names).to(device)
    opt = torch.optim.Adam(model.parameters(), lr = learning_rate)
    best_val_acc = 0.0

    for epoch in range(num_epoch):
        model.train()
        labels = y.to(device)
        logits= model(g, sub_data)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask].squeeze(-1)) 
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc = evaluate(g, labels, val_mask, model, sub_data)
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), 'best_model.pt')
        print(
                "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                    epoch, loss.item(), acc
                )
            )

    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    test_predictions = []
    test_labels = []
    labels = labels.to(device)

    with torch.no_grad():
        logits = model(g, sub_data)
        predictions = torch.argmax(logits[test_mask], dim=1)
        test_predictions.extend(predictions.tolist())
        test_labels.extend(labels[test_mask].tolist())
        report = classification_report(test_labels, test_predictions,  digits=5)
        if not os.path.exists("../report"):
            os.mkdir("../report")
        with open("../report/darpa_reports.txt", "a") as report_file:
            report_file.write(report)
        report_array = str.split(report)
        accuracy = float(report_array[25])
        recall = float(report_array[11])
        return accuracy, recall


if __name__ == "__main__":
    num_train_and_eval = 1
    num_epoch = 10
    learning_rate = 0.001
    accuracy_accumulator = []
    recall_accumulator = []

    '''graph_rel_names, _ = dgl.load_graphs('../data/darpa/darpa_graph/rel_names.dgl')
    graph_rel_names = graph_rel_names[0]
    rel_names = graph_rel_names.etypes'''

    loaded_graphs, _ = dgl.load_graphs("../data/darpa/darpa_graph/ta1-trace-e3-official-1.csv.dgl")
    loaded_graphs = loaded_graphs[0]
    g = loaded_graphs.to(device)

    rel_names = g.etypes

    for i in range(num_train_and_eval):
        accuracy, recall = darpa_train(i, g, rel_names, num_epoch, learning_rate)
        accuracy_accumulator.append(accuracy)
        recall_accumulator.append(recall)

    print(num_train_and_eval, " times train and evaluation report:")
    print("time\t\taccuracy\t\trecall")
    for i in range(num_train_and_eval):
        print(i, "\t\t", accuracy_accumulator[i], "\t\t", recall_accumulator[i])
    print("avg\t\t", np.mean(accuracy_accumulator), "\t\t", np.mean(recall_accumulator))