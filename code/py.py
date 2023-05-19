import os
import sys
import pandas as pd
import numpy as np
# print(os.getcwd())
data_path = '../data/'
base_file_name = 'true_labels.eng.'
files = ['testa', 'testb', 'train']

for f in files : 
    df = pd.read_csv(data_path + base_file_name + f + '.csv', sep=',', header=None)
    print(df.shape)
    # df2 = (df == 'I-PER').astype(int)
    # # print(df2)
    # df2.to_csv(data_path + base_file_name + f + '_binary.csv', sep=',', header=None, index=False)