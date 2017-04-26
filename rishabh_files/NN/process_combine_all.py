import pandas as pd
import numpy as np
import math
import os
import argparse

data_path = '/home/physics/btech/ph1140797/scratch/AHS_data/'

def intersect(a, b):
    return list(set(a) & set(b))

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def get_dataset(label):
	_1 = pd.read_csv(data_path + str(label) + '_AHS_COMB_diag_hotData.csv')
	_2 = pd.read_csv(data_path + str(label) + '_AHS_COMB_diag_col.csv')
	return(_1, _2)

print('Reading Data')
_1_list, _2_list = list(), list()
for label in states_list:
	_1, _2 = get_dataset(label)
	_1_list.append(_1)
	_2_list.append(_2)

print('Appending Data')
diagnosed_data = pd.concat(_1_list)
diagnosed_col = pd.concat(_2_list)

print(diagnosed_data.shape, diagnosed_col.shape)

print('Saving Data')
diagnosed_data.to_csv(data_path + 'All_states_COMB_diagHot.csv', index=False)
diagnosed_col.to_csv(data_path + 'All_states_COMB_col.csv', index=False)




