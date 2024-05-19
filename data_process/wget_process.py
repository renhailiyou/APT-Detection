import os
import json
import pandas as pd
import numpy as np
import torch
import datetime
import dgl
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# cf:date 边或节点在用户空间中被记录的时间 (string)
# cf:secctx Linux安全模块中定义的安全上下文 (string)
# entity
#   "prov:type":"process_memory",
#   "cf:date":"2018:10:06T15:52:05",
#   file -- 'cf:mode', 'cf:secctx'
#   process_memory -- 'cf:secctx'
#   mmaped_file -- 'cf:mode', 'cf:secctx',
#   socket -- 'cf:mode', 'cf:secctx',
#   path -- cf:pathname',
#   address -- 'cf:address'("type":"AF_INET", "host":"server-143-204-181-78.lhr50.r.cloudfront.net", "serv":"http")
#   link -- cf:mode', 'cf:secctx',
#   pipe -- 'cf:mode', 'cf:secctx',
#   argv -- 'cf:value',
#   iattr -- 'cf:mode',
#   block -- 'cf:mode', 'cf:secctx',
#   shm -- 'cf:mode'
# activity
#   "prov:type":"task",
#   "cf:date":"2018:10:06T15:52:05"
# used
#   "prov:type":"proc_read",
#   "cf:date":"2018:10:06T15:52:05",
#   "prov:entity":"",(entity)
#   "prov:activity":"",(activity)
# wasGeneratedBy
#   "prov:type":"clone_mem",
#   "cf:date":"2018:10:06T15:52:05",
#   "prov:activity":"",(activity)
#   "prov:entity":"",(entity)
# wasInformedBy
#   "prov:type":"terminate_task",
#   "cf:date":"2018:10:06T15:52:05"
#   "prov:informant":"",(activity)
#   "prov:informed":",(activity)
# wasDerivedFrom
#   "prov:type":"sh_read",
#   "cf:date":"2018:10:06T15:52:05",
#   "prov:usedEntity":"",(entity)
#   "prov:generatedEntity":"",(entity)

device = torch.device('cuda:0')

def save_rel_names():
    entity_attr = ["entity_type", "entity_secctx", "entity_mode", "entity_pathname", "entity_value"]
    entity_address = ["entity_address_type", "entity_address_host", "entity_address_serv"]
    entity_time = ["entity_year", "entity_month", "entity_day", "entity_hour", "entity_minute", "entity_second"]
    entity_connect_1 = ["entity_type", "entity_address_type"]
    entity_connect_2 = ["entity_type", "entity_day"]
    activity_time = ["activity_year", "activity_month", "activity_day", "activity_hour", "activity_minute", "activity_second"]
    activity_connect = ["activity_type", "activity_day"]
    edge_time = ["edge_year", "edge_month", "edge_day", "edge_hour", "edge_minute", "edge_second"]
    edge_connect_1 = ["edge_type", "edge_day"]

    graph_all_list = [
        entity_attr, entity_address, entity_time, entity_connect_1, entity_connect_2, 
        activity_time, activity_connect, edge_time, edge_connect_1
    ]
    graph_all = {}
    for i, j in enumerate(graph_all_list):
        graph_all[i] = j

    data_singel_graph_dict = {}
    for i, columns_to_convert in graph_all.items():
        for src in columns_to_convert:
            for dst in columns_to_convert:
                if src != dst:
                    data_singel_graph_dict.update({
                        (src, src + "_to_" + dst, dst): (torch.tensor([1], dtype = torch.int), torch.tensor([1], dtype = torch.int))
                    })
    data_singel_graph_dict.update({("entity_type", "entity_src_to_edge", "edge_type"): (torch.tensor([1], dtype = torch.int), torch.tensor([1], dtype = torch.int))})
    data_singel_graph_dict.update({("edge_type", "edge_to_entity_src", "entity_type"): (torch.tensor([1], dtype = torch.int), torch.tensor([1], dtype = torch.int))})
    data_singel_graph_dict.update({("activity_type", "activity_src_to_edge", "edge_type"): (torch.tensor([1], dtype = torch.int), torch.tensor([1], dtype = torch.int))})
    data_singel_graph_dict.update({("edge_type", "edge_to_activity_src", "activity_type"): (torch.tensor([1], dtype = torch.int), torch.tensor([1], dtype = torch.int))})
    data_singel_graph_dict.update({("entity_type", "entity_dst_to_edge", "edge_type"): (torch.tensor([1], dtype = torch.int), torch.tensor([1], dtype = torch.int))})
    data_singel_graph_dict.update({("edge_type", "edge_to_entity_dst", "entity_type"): (torch.tensor([1], dtype = torch.int), torch.tensor([1], dtype = torch.int))})
    data_singel_graph_dict.update({("activity_type", "activity_dst_to_edge", "edge_type"): (torch.tensor([1], dtype = torch.int), torch.tensor([1], dtype = torch.int))})
    data_singel_graph_dict.update({("edge_type", "edge_to_activity_dst", "activity_type"): (torch.tensor([1], dtype = torch.int), torch.tensor([1], dtype = torch.int))})
    hetero_graph = dgl.heterograph(data_singel_graph_dict).to(device)
    dgl.save_graphs('../data/wget/wget_graph/rel_names.dgl', [hetero_graph])

def raw_to_csv(dir_name):
    raw_path = dir_name + "raw/"
    save_path = dir_name + "wget_csv/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_column = ["entity/", "activity/", "edge/"]

    entity_column = ["prov:type", "cf:date", "cf:mode", "cf:secctx", "cf:pathname", "cf:value"]
    entity_column_csv = ["type", "date", "mode", "secctx", "pathname", "value", "address_type", "address_host", "address_serv"]

    activity_column = ["prov:type", "cf:date"]
    activity_column_csv = ["type", "date"]

    used_column = ["prov:type", "cf:date", "prov:entity", "prov:activity"]
    wasGeneratedBy_column = ["prov:type", "cf:date", "prov:activity", "prov:entity"]
    wasInformedBy_column = ["prov:type", "cf:date", "prov:informant", "prov:informed"]
    wasDerivedFrom_column = ["prov:type", "cf:date", "prov:usedEntity", "prov:generatedEntity"]
    edge_column_csv = ["type", "edge_type", "date", "src_type", "dst_type"]
    
    for file_name in tqdm(os.listdir(raw_path), desc='Processing'):
        if not os.path.isfile(raw_path + file_name):
            continue
        with open(raw_path + file_name, "r") as f:
            node_map = {}
            entity_list = []
            activity_list = []
            edge_list = []
            for line in f:
                try:
                    json_object = json.loads(line)
                except Exception as e:
                    print("Exception ({}) occurred when parsing a node in JSON:".format(e))
                    print(line)
                    exit(1)
                if "entity" in json_object:
                    entity = json_object["entity"]
                    for uid in entity:
                        if not uid in node_map:
                            if "prov:type" in entity[uid]:
                                node_map[uid] = entity[uid]["prov:type"]
                        row_list = []
                        for attr in entity_column:
                            if attr in entity[uid]:
                                if attr == "cf:address":
                                    for address_attr in entity[uid][attr]:
                                        row_list.append(entity[uid][attr][address_attr])
                                else:
                                    row_list.append(entity[uid][attr])
                            else:
                                row_list.append("NA")
                        if "cf:address" in entity[uid]:
                            for address_attr in entity[uid]["cf:address"]:
                                row_list.append(entity[uid]["cf:address"][address_attr])
                        else:
                            for i in range(3):
                                row_list.append("NA")
                        entity_list.append(row_list)
                if "activity" in json_object:
                    activity = json_object["activity"]
                    for uid in activity:
                        if not uid in node_map:
                            if "prov:type" in activity[uid]:
                                node_map[uid] = activity[uid]["prov:type"]
                        row_list = []
                        for attr in activity_column:
                            if attr in activity[uid]:
                                row_list.append(activity[uid][attr])
                            else:
                                row_list.append("NA")
                        activity_list.append(row_list)
            
            # 再次遍历
            f.seek(0, 0)
            for line in f:
                try:
                    json_object = json.loads(line)
                except Exception as e:
                    print("Exception ({}) occurred when parsing a node in JSON:".format(e))
                    print(line)
                    exit(1)
                if "used" in json_object:
                    used = json_object["used"]
                    for uid in used:
                        row_list = ["used"]
                        flag = True
                        for i, attr in enumerate(used_column):
                            if attr in used[uid]:
                                if i == 2 or i == 3:
                                    if used[uid][attr] not in node_map:
                                        flag = False
                                    else:
                                        row_list.append(node_map[used[uid][attr]])
                                else:
                                    row_list.append(used[uid][attr])
                            else:
                                row_list.append("NA")
                        if flag:
                            edge_list.append(row_list)
                if "wasGeneratedBy" in json_object:
                    wasGeneratedBy = json_object["wasGeneratedBy"]
                    for uid in wasGeneratedBy:
                        row_list = ["wasGeneratedBy"]
                        flag = True
                        for i, attr in enumerate(wasGeneratedBy_column):
                            if attr in wasGeneratedBy[uid]:
                                if i == 2 or i == 3:
                                    if wasGeneratedBy[uid][attr] not in node_map:
                                        flag = False
                                    else:
                                        row_list.append(node_map[wasGeneratedBy[uid][attr]])
                                else:
                                    row_list.append(wasGeneratedBy[uid][attr])
                            else:
                                row_list.append("NA")
                        if flag:
                            edge_list.append(row_list)
                if "wasInformedBy" in json_object:
                    wasInformedBy = json_object["wasInformedBy"]
                    for uid in wasInformedBy:
                        row_list = ["wasInformedBy"]
                        flag = True
                        for i, attr in enumerate(wasInformedBy_column):
                            if attr in wasInformedBy[uid]:
                                if i == 2 or i == 3:
                                    if wasInformedBy[uid][attr] not in node_map:
                                        flag = False
                                    else:
                                        row_list.append(node_map[wasInformedBy[uid][attr]])
                                else:
                                    row_list.append(wasInformedBy[uid][attr])
                            else:
                                row_list.append("NA")
                        if flag:
                            edge_list.append(row_list)
                if "wasDerivedFrom" in json_object:
                    wasDerivedFrom = json_object["wasDerivedFrom"]
                    for uid in wasDerivedFrom:
                        row_list = ["wasDerivedFrom"]
                        flag = True
                        for i, attr in enumerate(wasDerivedFrom_column):
                            if attr in wasDerivedFrom[uid]:
                                if i == 2 or i == 3:
                                    if wasDerivedFrom[uid][attr] not in node_map:
                                        flag = False
                                    else:
                                        row_list.append(node_map[wasDerivedFrom[uid][attr]])
                                else:
                                    row_list.append(wasDerivedFrom[uid][attr])
                            else:
                                row_list.append("NA")
                        if flag:
                            edge_list.append(row_list)

            for i in save_column:
                if not os.path.exists(save_path + i):
                    os.mkdir(save_path + i)
            entity_DataFrame = pd.DataFrame(columns = entity_column_csv, data = entity_list)
            entity_DataFrame.to_csv(save_path + "entity/" + file_name[:-4] + "-entity.csv")
            activity_DataFrame = pd.DataFrame(columns = activity_column_csv, data = activity_list)
            activity_DataFrame.to_csv(save_path + "activity/" + file_name[:-4] + "-activity.csv")
            edge_DataFrame = pd.DataFrame(columns = edge_column_csv, data = edge_list)
            edge_DataFrame.to_csv(save_path + "edge/" + file_name[:-4] + "-edge.csv")

def my_date_parser(data_str):
    return datetime.datetime.strptime(data_str, "%Y:%m:%dT%H:%M:%S")

def csv_process(dir_name):
    raw_path = dir_name + "raw/"
    csv_path = dir_name + "wget_csv/"
    save_path = dir_name + "wget_csv_processed/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    csv_column = ["entity", "activity", "edge"]
    all_names = [
        [
            "entity_type", "entity_date", "entity_mode", "entity_secctx", 
            "entity_pathname", "entity_value", "entity_address_type", 
            "entity_address_host", "entity_address_serv"
        ], 
        ["activity_type", "activity_date"], 
        ["type", "edge_type", "edge_date", "src_type", "dst_type"]
    ]
    
    for file_name in tqdm(os.listdir(raw_path), desc='Processing'):
        if not os.path.isfile(raw_path + file_name):
            continue
        for i, j in enumerate(csv_column):
            data = pd.read_csv(
                csv_path + j +"/" + file_name[:-4] + "-" + j + ".csv",
                sep = ',', header = 0, index_col = 0,
                names = all_names[i],
                parse_dates = [j + "_date"], date_parser = my_date_parser
            )
            data[j + "_date"] = pd.to_datetime(data[j + "_date"])
            data[j + "_year"] = data[j + "_date"].dt.year
            data[j + "_month"] = data[j + "_date"].dt.month
            data[j + "_day"] = data[j + "_date"].dt.day
            data[j + "_hour"] = data[j + "_date"].dt.hour
            data[j + "_minute"] = data[j + "_date"].dt.minute
            data[j + "_second"] = data[j + "_date"].dt.second
            if not os.path.exists(save_path + j + "/"):
                os.mkdir(save_path + j + "/")
            data.to_csv(save_path + j + "/" + file_name[:-4] + "-" + j + ".csv")

def del_nan(A,B):
    array_A = np.array(A)
    array_B = np.array(B)
    nan_mask = np.isnan(array_A) | np.isnan(array_B)
    valid_indices = ~nan_mask
    return array_A[valid_indices], array_B[valid_indices]

def find_edge_type(edge):
    edge = np.array(edge)
    used_index = np.where(edge == "used")
    wasGeneratedBy_index = np.where(edge == "wasGeneratedBy")
    wasInformedBy_index = np.where(edge == "wasInformedBy")
    wasDerivedFrom_index = np.where(edge == "wasDerivedFrom")
    return used_index, wasGeneratedBy_index, wasInformedBy_index, wasDerivedFrom_index

def complete_graph_edge(
    data_dict, graph_all, data_list, 
    encoding_to_src_type, encoding_to_dst_type,
    entity_type_to_encoding, activity_type_to_encoding
):
    for i, columns_to_convert in graph_all.items():
        for src in columns_to_convert:
            for dst in columns_to_convert:
                if src != dst:
                    new_list_A, new_list_B = del_nan(data_list[src], data_list[dst])
                    data_dict.update({
                        (src, src + "_to_" + dst, dst): 
                        (torch.tensor(new_list_A, dtype = torch.int), torch.tensor(new_list_B, dtype = torch.int))
                    })
    list_used, list_wasGeneratedBy, list_wasInformedBy, list_wasDerivedFrom = find_edge_type(
        data_list["type"]
    )
    used_src = np.array(data_list["src_type"])[list_used]
    used_dst = np.array(data_list["dst_type"])[list_used]
    wasGeneratedBy_src = np.array(data_list["src_type"])[list_wasGeneratedBy]
    wasGeneratedBy_src = np.array(data_list["src_type"])[list_wasGeneratedBy]
    wasGeneratedBy_dst = np.array(data_list["dst_type"])[list_wasGeneratedBy]
    wasInformedBy_src = np.array(data_list["src_type"])[list_wasInformedBy]
    wasInformedBy_dst = np.array(data_list["dst_type"])[list_wasInformedBy]
    wasDerivedFrom_src = np.array(data_list["src_type"])[list_wasDerivedFrom]
    wasDerivedFrom_dst = np.array(data_list["dst_type"])[list_wasDerivedFrom]

    used_edge_type = np.array(data_list["edge_type"])[list_used].tolist()
    wasGeneratedBy_edge_type = np.array(data_list["edge_type"])[list_wasGeneratedBy].tolist()
    wasInformedBy_edge_type = np.array(data_list["edge_type"])[list_wasInformedBy].tolist()
    wasDerivedFrom_edge_type = np.array(data_list["edge_type"])[list_wasDerivedFrom].tolist()

    used_src = pd.Series(used_src).map(encoding_to_src_type).map(entity_type_to_encoding).tolist()
    wasGeneratedBy_src = pd.Series(wasGeneratedBy_src).map(encoding_to_src_type).map(activity_type_to_encoding).tolist()
    wasInformedBy_src = pd.Series(wasInformedBy_src).map(encoding_to_src_type).map(activity_type_to_encoding).tolist()
    wasDerivedFrom_src = pd.Series(wasDerivedFrom_src).map(encoding_to_src_type).map(entity_type_to_encoding).tolist()
    
    used_dst = pd.Series(used_dst).map(encoding_to_dst_type).map(activity_type_to_encoding).tolist()
    wasGeneratedBy_dst = pd.Series(wasGeneratedBy_dst).map(encoding_to_dst_type).map(entity_type_to_encoding).tolist()
    wasInformedBy_dst = pd.Series(wasInformedBy_dst).map(encoding_to_dst_type).map(activity_type_to_encoding).tolist()
    wasDerivedFrom_dst = pd.Series(wasDerivedFrom_dst).map(encoding_to_dst_type).map(entity_type_to_encoding).tolist()

    # src: entity_type <-> edge_type
    entity_src = used_src + wasDerivedFrom_src
    entity_src_to_edge_type = used_edge_type + wasDerivedFrom_edge_type
    new_list_A, new_list_B = del_nan(entity_src, entity_src_to_edge_type)
    data_dict.update({
        ("entity_type", "entity_src_to_edge", "edge_type"):
        (torch.tensor(new_list_A, dtype=torch.int), torch.tensor(new_list_B, dtype=torch.int))
    })
    data_dict.update({
        ("edge_type", "edge_to_entity_src", "entity_type"):
        (torch.tensor(new_list_B, dtype=torch.int), torch.tensor(new_list_A, dtype=torch.int))
    })
    # src: activity_type <-> edge_type
    activity_src = wasGeneratedBy_src + wasInformedBy_src
    activity_src_to_edge_type = wasGeneratedBy_edge_type + wasInformedBy_edge_type
    new_list_A, new_list_B = del_nan(activity_src, activity_src_to_edge_type)
    data_dict.update({
        ("activity_type", "activity_src_to_edge", "edge_type"):
        (torch.tensor(new_list_A, dtype=torch.int), torch.tensor(new_list_B, dtype=torch.int))
    })
    data_dict.update({
        ("edge_type", "edge_to_activity_src", "activity_type"):
        (torch.tensor(new_list_B, dtype=torch.int), torch.tensor(new_list_A, dtype=torch.int))
    })
    # dst: edge_type <-> entity_type
    entity_dst = wasGeneratedBy_dst + wasDerivedFrom_dst
    entity_dst_to_edge_type = wasGeneratedBy_edge_type + wasDerivedFrom_edge_type
    new_list_A, new_list_B = del_nan(entity_dst, entity_dst_to_edge_type)
    data_dict.update({
        ("entity_type", "entity_dst_to_edge", "edge_type"):
        (torch.tensor(new_list_A, dtype=torch.int), torch.tensor(new_list_B, dtype=torch.int))
    })
    data_dict.update({
        ("edge_type", "edge_to_entity_dst", "entity_type"):
        (torch.tensor(new_list_B, dtype=torch.int), torch.tensor(new_list_A, dtype=torch.int))
    })
    # dst:edge_type <-> activity_type
    activity_dst = used_dst + wasInformedBy_dst
    activity_dst_to_edge_type = used_edge_type + wasInformedBy_edge_type
    new_list_A, new_list_B = del_nan(activity_dst, activity_dst_to_edge_type)
    data_dict.update({
        ("activity_type", "activity_dst_to_edge", "edge_type"):
        (torch.tensor(new_list_A, dtype=torch.int), torch.tensor(new_list_B, dtype=torch.int))
    })
    data_dict.update({
        ("edge_type", "edge_to_activity_dst", "activity_type"):
        (torch.tensor(new_list_B, dtype=torch.int), torch.tensor(new_list_A, dtype=torch.int))
    })

def build_single_graph(data_dict, data, dir_name, file_name):
    json_save_path = dir_name + "wget_column_encoding/"
    if not os.path.exists(json_save_path):
        os.mkdir(json_save_path)
    graph_save_path = dir_name + "wget_graph/"
    if not os.path.exists(graph_save_path):
        os.mkdir(graph_save_path)
    column_encoding = {}
    feature_lists = {}
    data_list = {}
    column_encoding_json = {}
    encoding_to_src_type = {}
    encoding_to_dst_type = {}
    columns_to_convert = [
        [
            "entity_type", "entity_mode", "entity_secctx", 
            "entity_pathname", "entity_value", "entity_address_type", 
            "entity_address_host", "entity_address_serv", 
            "entity_year", "entity_month", "entity_day", "entity_hour", "entity_minute", "entity_second"
        ], 
        [
            "activity_type", 
            "activity_year", "activity_month", "activity_day", "activity_hour", "activity_minute", "activity_second"
        ], 
        [
            "edge_type", "src_type", "dst_type", 
            "edge_year", "edge_month", "edge_day", "edge_hour", "edge_minute", "edge_second"
        ]
    ]
    columns_feature_to_convert = [
        "entity_type", "entity_mode", "entity_secctx", 
        "entity_pathname", "entity_value", "entity_address_type", 
        "entity_address_host", "entity_address_serv", 
        "entity_year", "entity_month", "entity_day", "entity_hour", "entity_minute", "entity_second", 
        "activity_type", 
        "activity_year", "activity_month", "activity_day", "activity_hour", "activity_minute", "activity_second", 
        "edge_type","edge_year", "edge_month", "edge_day", "edge_hour", "edge_minute", "edge_second"
    ]
    # 对columns_to_convert中的列编码
    for i in range(3):
        for col in columns_to_convert[i]:
            unique_values = data[i][col].dropna().unique()
            encoding = {value: j for j, value in enumerate(unique_values)}
            feature_lists[col] = [value for value in unique_values]
            column_encoding[col] = encoding
            column_encoding_json[col] = {str(value): j for value, j in encoding.items()}
            if col == "src_type":
                encoding_to_src_type = {j: value for value, j in encoding.items()}
            if col == "entity_type":
                entity_type_to_encoding = encoding
            if col == "dst_type":
                encoding_to_dst_type = {j: value for value, j in encoding.items()}
            if col == "activity_type":
                activity_type_to_encoding = encoding
        for col, encoding in column_encoding.items():
            data[i][col] = data[i][col].map(encoding)
        column_encoding = {}
    
    with open(json_save_path + file_name + ".json", "w") as file:
        json.dump(column_encoding_json, file)

    for i in range(3):
        for col in columns_to_convert[i]:
            data_list[col] = data[i][col].tolist()
    data_list["type"] = data[2]["type"].tolist()

    # 联通分量
    entity_attr = ["entity_type", "entity_secctx", "entity_mode", "entity_pathname", "entity_value"]
    entity_address = ["entity_address_type", "entity_address_host", "entity_address_serv"]
    entity_time = ["entity_year", "entity_month", "entity_day", "entity_hour", "entity_minute", "entity_second"]
    entity_connect_1 = ["entity_type", "entity_address_type"]
    entity_connect_2 = ["entity_type", "entity_day"]
    activity_time = ["activity_year", "activity_month", "activity_day", "activity_hour", "activity_minute", "activity_second"]
    activity_connect = ["activity_type", "activity_day"]
    edge_time = ["edge_year", "edge_month", "edge_day", "edge_hour", "edge_minute", "edge_second"]
    edge_connect_1 = ["edge_type", "edge_day"]

    graph_all_list = [
        entity_attr, entity_address, entity_time, entity_connect_1, entity_connect_2, 
        activity_time, activity_connect, edge_time, edge_connect_1
    ]
    graph_all = {}
    for i, j in enumerate(graph_all_list):
        graph_all[i] = j
    
    complete_graph_edge(
        data_dict, graph_all, data_list, 
        encoding_to_src_type, encoding_to_dst_type,
        entity_type_to_encoding, activity_type_to_encoding
    )
    single_hetero_graph = dgl.heterograph(data_dict).to(device)

    for column in columns_feature_to_convert:
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
        single_hetero_graph.nodes[column].data['feat'] = torch.cat(semantic_vectors, dim=0).to(device)# 768维的特征向量
    dgl.save_graphs(graph_save_path + file_name + ".dgl", [single_hetero_graph])

def csv_to_graph(dir_name):
    raw_path = dir_name + "raw/"
    csv_path = dir_name + "wget_csv_processed/"

    csv_column = ["entity", "activity", "edge"]

    for file_name in tqdm(os.listdir(raw_path), desc='Processing'):
        if not os.path.isfile(raw_path + file_name):
            continue
        data = []
        for j in csv_column:
            sub_data = pd.read_csv(
                csv_path + j +"/" + file_name[:-4] + "-" + j + ".csv",
                sep = ',', header = 'infer', index_col = 0,
            )
            data.append(sub_data)
        single_graph_dict = {}
        build_single_graph(single_graph_dict, data, dir_name, file_name[:-4])

if __name__ == "__main__":
    dir_name = "../data/wget/"

    # 将CamFlow格式转为csv格式
    # raw_to_csv(dir_name)

    # 预处理日期并改列名
    # csv_process(dir_name)

    # 保存所有变关系
    save_rel_names()

    # 从csv格式中建图并保存
    # csv_to_graph(dir_name)