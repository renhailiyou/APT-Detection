import pandas as pd

data_1 = pd.read_csv("../data/darpa/darpa_csv/trace-benign.csv")
data_2 = pd.read_csv("../data/darpa/darpa_csv/trace-malicious.csv")
sub_data_1 = data_1.sample(n = 3, random_state = 1).reset_index(drop = True)
sub_data_2 = data_2.sample(n = 3, random_state = 1).reset_index(drop = True)
data_list = [sub_data_1, sub_data_2]
df_merged = pd.concat(data_list, axis = 0, ignore_index = True)
print(df_merged)