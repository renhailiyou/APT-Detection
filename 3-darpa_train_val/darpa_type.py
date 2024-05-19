import argparse
import json
import os
import random
import re

from tqdm import tqdm
import networkx as nx
import pickle as pkl

raw_path = '../data/darpa/trace/'

type_name = {}

'''for file_name in tqdm(os.listdir(raw_path), desc='Processing'):
    if not os.path.isfile(raw_path + file_name):
        continue
with open(raw_path + file_name, "r") as f:'''
with open('../data/darpa/trace/ta1-trace-e3-official-1.json', "r") as f:
    for line in f:
        try:
            json_object = json.loads(line)
        except Exception as e:
            print("Exception ({}) occurred when parsing a node in JSON:".format(e))
            print(line)
            exit(1)
        datum = json_object["datum"]
        for i in datum:
            attr_dict = []
            for attr in datum[i]:
                attr_dict.append(attr)
            if i not in type_name:
                type_name[i] = attr_dict
for i in type_name:
    print(i, type_name[i])