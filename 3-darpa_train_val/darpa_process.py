import os
import json
import pandas as pd
import numpy as np
import torch
import datetime
import dgl
import re
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

device = torch.device('cuda:0')

# 匹配的正则表达式
pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')

pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_process_name = re.compile(r'map\":\{\"name\":\"(.*?)\"')
pattern_file_path = re.compile(r'map\":\{\"path\":\"(.*?)\"')
pattern_remote_address = re.compile(r'remoteAddress\":\"(.*?)\"')
pattern_memory_size = re.compile(r'size\":\{\"long\":\"(.*?)\"')

def raw_to_csv(dir_name):
    raw_path = dir_name + "raw/"
    file_name = "ta1-trace-e3-official-1.json"
    malicious_file = "trace.txt"
    save_path = dir_name + "darpa_csv/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    column_csv = ["src_type", "src_process_name", "src_file_path", "src_remote_address", "src_memory_size", 
                  "edge_type", 
                  "dst_type", "dst_process_name", "dst_file_path", "dst_remote_address", "dst_memory_size", 
                  "label"]

    m_f = open(raw_path + malicious_file, 'r')
    malicious_entities = set()
    for l in m_f.readlines():
        # 移除字符串开头和末尾的空白字符
        malicious_entities.add(l.lstrip().rstrip())
    m_f.close()
    id_node_attr_map = {}
    csv_list = []
    csv_benign_list = []
    csv_malicious_list = []
    with open(raw_path + file_name, "r") as f:
        # 第一次遍历获取节点信息
        for line in tqdm(f, desc = "Get node information"):
            if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line: continue
            if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line: continue
            if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line: continue
            row_list = []
            if len(pattern_uuid.findall(line)) == 0: print(line)
            uuid = pattern_uuid.findall(line)[0]
            # UnnamedPipeObject -
            # SrcSinkObject - type
            # Principal - type
            # NetFlowObject - remoteaddress
            # MemoryObject - size
            # Subject - type parentSubject
            # FileObject - type filepath
            if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                subject_type = 'UnnamedPipeObject'
                row_list.append(subject_type)
                for i in range(4):
                    row_list.append("NA")
            elif 'com.bbn.tc.schema.avro.cdm18.SrcSinkObject' in line or 'com.bbn.tc.schema.avro.cdm18.Principal' in line:
                subject_type = pattern_type.findall(line)
                row_list.append(subject_type[0])
                for i in range(4):
                    row_list.append("NA")
            elif 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                subject_type = 'NetFlowObject'
                row_list.append(subject_type)
                for i in range(2):
                    row_list.append("NA")
                remote_address = pattern_remote_address.findall(line)
                row_list.append(remote_address[0])
                row_list.append("NA")
            elif 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                subject_type = 'MemoryObject'
                row_list.append(subject_type)
                for i in range(3):
                    row_list.append("NA")
                if len(pattern_memory_size.findall(line)) > 0:
                    row_list.append(pattern_memory_size.findall(line)[0])
                else:
                    row_list.append("NA")
            elif 'com.bbn.tc.schema.avro.cdm18.Subject' in line:
                subject_type = pattern_type.findall(line)
                row_list.append(subject_type[0])
                if subject_type == 'SUBJECT_PROCESS' and len(pattern_process_name.findall(line)) > 0:
                    row_list.append(pattern_process_name.findall(line)[0])
                else:
                    row_list.append("NA")
                for i in range(3):
                    row_list.append("NA")
            elif 'com.bbn.tc.schema.avro.cdm18.FileObject' in line:
                subject_type = pattern_type.findall(line)
                row_list.append(subject_type[0])
                row_list.append("NA")
                file_path = pattern_file_path.findall(line)
                row_list.append(file_path[0])
                for i in range(2):
                    row_list.append("NA")
            id_node_attr_map[uuid] = row_list
        # 再次遍历获取边
        f.seek(0, 0)
        for line in tqdm(f, desc = "Get full information"):
            if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                edge_type = pattern_type.findall(line)[0]
                # 源节点
                src_uuid = pattern_src.findall(line)
                if len(src_uuid) == 0: continue
                src_uuid = src_uuid[0]
                if not src_uuid in id_node_attr_map:
                    continue
                src_attr_list = id_node_attr_map[src_uuid]
                # 目标节点1
                dst_uuid_1 = pattern_dst1.findall(line)
                if len(dst_uuid_1) > 0 and dst_uuid_1[0] != 'null':
                    dst_uuid_1 = dst_uuid_1[0]
                    if not dst_uuid_1 in id_node_attr_map:
                        continue
                    dst_attr_list_1 = id_node_attr_map[dst_uuid_1]
                    full_row_list = []
                    for i in src_attr_list:
                        full_row_list.append(i)
                    full_row_list.append(edge_type)
                    for i in dst_attr_list_1:
                        full_row_list.append(i)
                    if src_uuid in malicious_entities or dst_uuid_1 in malicious_entities:
                        full_row_list.append(1)
                        csv_malicious_list.append(full_row_list)
                    else:
                        full_row_list.append(0)
                        csv_benign_list.append(full_row_list)
                    csv_list.append(full_row_list)
                # 目标节点2
                dst_uuid_2 = pattern_dst1.findall(line)
                if len(dst_uuid_2) > 0 and dst_uuid_2[0] != 'null':
                    dst_uuid_2 = dst_uuid_2[0]
                    if not dst_uuid_2 in id_node_attr_map:
                        continue
                    dst_attr_list_2 = id_node_attr_map[dst_uuid_2]
                    full_row_list = []
                    for i in src_attr_list:
                        full_row_list.append(i)
                    full_row_list.append(edge_type)
                    for i in dst_attr_list_2:
                        full_row_list.append(i)
                    if src_uuid in malicious_entities or dst_uuid_2 in malicious_entities:
                        full_row_list.append(1)
                        csv_malicious_list.append(full_row_list)
                    else:
                        full_row_list.append(0)
                        csv_benign_list.append(full_row_list)
                    csv_list.append(full_row_list)
    f.close()
    darpa_DataFrame = pd.DataFrame(columns = column_csv, data = csv_list)
    darpa_DataFrame.to_csv(save_path + "ta1-trace-e3-official-1.csv")
    darpa_DataFrame = pd.DataFrame(columns = column_csv, data = csv_benign_list)
    darpa_DataFrame.to_csv(save_path + "trace-benign.csv")
    darpa_DataFrame = pd.DataFrame(columns = column_csv, data = csv_malicious_list)
    darpa_DataFrame.to_csv(save_path + "trace-malicious.csv")

def del_nan(A,B):
    array_A = np.array(A)
    array_B = np.array(B)
    nan_mask = np.isnan(array_A) | np.isnan(array_B)
    valid_indices = ~nan_mask
    return array_A[valid_indices], array_B[valid_indices]

def complete_graph_edge(data_dict, graph_all, data_list):
    for i, columns_to_convert in graph_all.items():
        for src in columns_to_convert:
            for dst in columns_to_convert:
                if src != dst:
                    new_list_A, new_list_B = del_nan(data_list[src], data_list[dst])
                    if len(new_list_A) != 0:
                        data_dict.update({
                            (src, src + "_to_" + dst, dst): 
                            (torch.tensor(new_list_A, dtype = torch.int), torch.tensor(new_list_B, dtype = torch.int))
                        })

def build_single_graph(data_dict, data, dir_name, file_name):
    json_save_path = dir_name + "darpa_column_encoding/"
    if not os.path.exists(json_save_path):
        os.mkdir(json_save_path)
    graph_save_path = dir_name + "darpa_graph/"
    if not os.path.exists(graph_save_path):
        os.mkdir(graph_save_path)
    column_encoding = {}
    feature_lists = {}
    data_list = {}
    column_encoding_json = {}
    columns_to_convert = [
        "src_type", "src_process_name", "src_file_path", "src_remote_address", "src_memory_size", 
        "edge_type", 
        "dst_type", "dst_process_name", "dst_file_path", "dst_remote_address", "dst_memory_size", 
        "label"
    ]
    columns_feature_to_convert = [
        "src_type", "src_process_name", "src_file_path", "src_remote_address", "src_memory_size", 
        "edge_type", 
        "dst_type", "dst_process_name", "dst_file_path", "dst_remote_address", "dst_memory_size", 
    ]
    # 对columns_to_convert中的列编码
    for col in columns_to_convert:
        unique_values = data[col].dropna().unique()
        encoding = {value: j for j, value in enumerate(unique_values)}
        feature_lists[col] = [value for value in unique_values]
        column_encoding[col] = encoding
        column_encoding_json[col] = {str(value): j for value, j in encoding.items()}
        
    for col, encoding in column_encoding.items():
        data[col] = data[col].map(encoding)
    
    with open(json_save_path + file_name + ".json", "w") as file:
        json.dump(column_encoding_json, file)

    for col in columns_to_convert:
        data_list[col] = data[col].tolist()

    # 联通分量
    src_1 = ["src_type", "src_process_name"]
    src_2 = ["src_type", "src_file_path"]
    src_3 = ["src_type", "src_remote_address"]
    src_4 = ["src_type", "src_memory_size"]
    dst_1 = ["dst_type", "dst_process_name"]
    dst_2 = ["dst_type", "dst_file_path"]
    dst_3 = ["dst_type", "dst_remote_address"]
    dst_4 = ["dst_type", "dst_memory_size"]
    src_edge = ["src_type", "edge_type"]
    dst_edge = ["dst_type", "edge_type"]
    
    graph_all_list = [
        src_1, src_2, src_3, src_4, 
        dst_1, dst_2, dst_3, dst_4, 
        src_edge, dst_edge
    ]
    graph_all = {}
    for i, j in enumerate(graph_all_list):
        graph_all[i] = j
    
    complete_graph_edge(data_dict, graph_all, data_list)

    # 建图
    single_hetero_graph = dgl.heterograph(data_dict).to(device)

    for column in columns_feature_to_convert:
        feature_list = feature_lists[column]
        feature_list_strings = list(map(str, feature_list))
        model_name = "../bert-base-uncased"
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
        if len(semantic_vectors) != 0:
            single_hetero_graph.nodes[column].data['feat'] = torch.cat(semantic_vectors, dim=0).to(device)# 768维的特征向量
    dgl.save_graphs(graph_save_path + file_name + ".dgl", [single_hetero_graph])

def csv_to_graph(dir_name):
    csv_path = dir_name + "darpa_csv/"
    for file_name in tqdm(os.listdir(csv_path), desc = "Processing"):
        data = []
        data = pd.read_csv(csv_path + file_name, sep = ',', header = 'infer', index_col = 0)
        single_graph_dict = {}
        build_single_graph(single_graph_dict, data, dir_name, file_name)

if __name__ == "__main__":
    dir_name = "../data/darpa/"

    # 将darap数据集json格式转为csv格式
    # raw_to_csv(dir_name)

    # 从csv格式中建图并保存
    csv_to_graph(dir_name)