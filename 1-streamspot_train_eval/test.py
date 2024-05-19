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
graph_rel_names, _ = dgl.load_graphs('../data/streamspot/single_graph/rel_names.dgl')
graph_rel_names = graph_rel_names[0]
rel_names = graph_rel_names.etypes
print(rel_names)