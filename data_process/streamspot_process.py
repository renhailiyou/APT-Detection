import pandas as pd
import dgl
import torch
import math
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer, BertModel
import json
import networkx as nx
from tqdm import tqdm

def save_rel_names(map_edge):
    data_single_graph_dict = {}
    for _, edge_type in map_edge.items():
        data_single_graph_dict.update({
            ('Src_Type', 'edge_' + edge_type, 'Dst_Type'):
            (
                torch.tensor([1], dtype=torch.int),
                torch.tensor([1], dtype=torch.int)
            )
        })
        data_single_graph_dict.update({
            ('Dst_Type', 'edge_be_' + edge_type + 'ed', 'Src_Type'):
            (
                torch.tensor([1], dtype=torch.int),
                torch.tensor([1], dtype=torch.int)
            )
        })
    hetero_graph = dgl.heterograph(data_single_graph_dict).to(device)
    dgl.save_graphs('../data/streamspot/single_graph/rel_names.dgl', [hetero_graph])

def complete_graph_edge(data_single_graph_dict, data_single_graph_list, edge_list):
    data_single_graph_edge_type_array = np.array(data_single_graph_list['Edge_Type'])
    data_single_graph_src_type_array = np.array(data_single_graph_list['Src_Type'])
    data_single_graph_dst_type_array = np.array(data_single_graph_list['Dst_Type'])
    for edge_type in edge_list:
        valid_index = np.where(data_single_graph_edge_type_array == edge_type, True, False)
        data_single_graph_dict.update({
            ('Src_Type', 'edge_' + map_edge[edge_type], 'Dst_Type'):
            (
                torch.tensor(data_single_graph_src_type_array[valid_index], dtype=torch.int),
                torch.tensor(data_single_graph_dst_type_array[valid_index], dtype=torch.int)
            )
        })
        ### 异构图必须是双向边
        data_single_graph_dict.update({
            ('Dst_Type', 'edge_be_' + map_edge[edge_type] + 'ed', 'Src_Type'):
            (
                torch.tensor(data_single_graph_dst_type_array[valid_index], dtype=torch.int),
                torch.tensor(data_single_graph_src_type_array[valid_index], dtype=torch.int)
            )
        })


device = torch.device('cuda:0')

NUM_GRAPHS = 600
map_node_1 = {
    'process': 'a', 'thread': 'b', 'file': 'c',
    'MAP_ANONYMOUS': 'd', 'NA': 'e', 'stdin': 'f',
    'stdout': 'g', 'stderr': 'h',
}
map_node_2 = {
    'a': 1, 'b': 2, 'c': 3,
    'd': 4, 'e': 5, 'f': 6,
    'g': 7, 'h':8
}
map_edge = {
    'i': 'accept', 'j': 'access', 'k': 'bind',
    'l': 'chmod', 'm': 'clone', 'n': 'close',
    'o': 'connect', 'p': 'execve', 'q': 'fstat',
    'r': 'ftruncate', 's': 'listen', 't': 'mmap2',
    'u': 'open', 'v': 'read', 'w': 'recv',
    'x': 'recvfrom', 'y': 'recvmsg', 'z': 'send',
    'A': 'sendmsg', 'B': 'sendto', 'C': 'stat',
    'D': 'truncate', 'E': 'unlink', 'F': 'waitpid',
    'G': 'write', 'H': 'writev'
}
columns_needed = ['Src_Type', 'Dst_Type', 'Edge_Type']
columns_to_convert = ['Src_Type', 'Dst_Type']

data_all_graph = pd.read_csv(
    "../data/streamspot/all.tsv", sep='\t', header=None,
    names=['Src_Id', 'Src_Type', 'Dst_Id', 'Dst_Type', 'Edge_Type', 'G_Id']
)

save_rel_names(map_edge)

for i in range(NUM_GRAPHS):
    data_single_graph = data_all_graph[data_all_graph['G_Id'] == i]
    column_encoding = {}
    column_encoding_json = {}
    feature_lists = {}
    for column in columns_to_convert:
        unique_values = data_single_graph[column].unique()
        encoding = {value: i for i, value in enumerate(unique_values)}
        feature_lists[column] = [value for value in unique_values]
        column_encoding[column] = encoding
        column_encoding_json[column] = {str(value): i for value, i in encoding.items()}

    for column, encoding in column_encoding.items():
        data_single_graph[column] = data_single_graph[column].map(encoding)

    with open('../data/streamspot/column_encoding/column_encoding_' + str(i) + '.json', 'w') as file:
        json.dump(column_encoding_json, file)

    data_single_graph_list = {column: data_single_graph[column].tolist() for column in columns_needed}
    edge_list = {value for value, _ in map_edge.items()}
    data_single_graph_dict = {}

    complete_graph_edge(data_single_graph_dict, data_single_graph_list, edge_list)
    hetero_graph = dgl.heterograph(data_single_graph_dict).to(device)

    for column in columns_to_convert:
        feature_list = feature_lists[column]
        feature_list_strings = list(map(str, feature_list))
        model_name = r"D:/Code/Python/BehaviorStructure/crime_detection/bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.to(device)
        semantic_vectors = []
        for value in feature_list_strings:
            encoded_text = tokenizer(value, return_tensors="pt").to(device)# 返回pytorch类型的tensor
            with torch.no_grad():
                outputs = model(**encoded_text)
                semantic_vector = outputs.last_hidden_state.mean(dim=1).to(device)
                semantic_vectors.append(semantic_vector)
        hetero_graph.nodes[column].data['feat'] = torch.cat(semantic_vectors, dim=0).to(device)# 768维的特征向量
        # print('column', column, hetero_graph.nodes[column].data['feat'].shape)
    dgl.save_graphs('../data/streamspot/single_graph/streamspot_' + str(i) + '.dgl', [hetero_graph])