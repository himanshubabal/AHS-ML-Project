import pandas as pd
import os
import argparse

from ..variables import data_path, data_scratch

data_path = data_scratch
states_list = [5, 8, 9, 10, 18, 20, 21, 22, 23]

def intersect(a, b):
    return list(set(a) & set(b))

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def get_dataset(label):
	_1 = pd.read_csv(data_path + str(label) + '/' + str(label) + '_AHS_COMB_diag_hotData.csv')
	_2 = pd.read_csv(data_path + str(label) + '/' + str(label) + '_AHS_COMB_diag_col.csv')
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

path = data_path + 'all/' 
if not os.path.exists(path):
	os.makedirs(path)

diagnosed_data.to_csv(path + 'All_states_COMB_diagHot.csv', index=False)
diagnosed_col.to_csv(path + 'All_states_COMB_col.csv', index=False)




