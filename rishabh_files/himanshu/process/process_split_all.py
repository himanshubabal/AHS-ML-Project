import pandas as pd
import numpy as np
import os
import argparse

import tensorflow as tf
from keras.utils.np_utils import to_categorical
from ..variables import data_path, data_scratch

data_path = data_scratch + 'all/'
 
if not os.path.exists(data_path):
	os.makedirs(data_path)

diag_hot_csv_name = 'All_states_COMB_diagHot.csv'
diag_col_csv_name = 'All_states_COMB_col.csv'

def replace_labes(label_data):
	    dict_map = {1.0 : 1.0, 2.0 : 2.0, 3.0 : 3.0, 7.0 : 4.0, 9.0 : 5.0,
	                19.0 : 6.0, 21.0 : 7.0, 99.0 : 7.0}
	    for i in range(len(label_data)):
	        if label_data[i] in dict_map:
	            label_data[i] = dict_map[label_data[i]]-1
	        else :
	            label_data[i] = 0.0
	    return label_data

def check_unnamed(dataframe):
	if 'Unnamed: 0' in list(dataframe):
		dataframe = dataframe.drop('Unnamed: 0',axis=1,errors='ignore')
	return dataframe


def split_data():
	diagnosed_data = pd.read_csv(data_path + diag_hot_csv_name)
	diagnosed_col = pd.read_csv(data_path + diag_col_csv_name)

	diagnosed_col = check_unnamed(diagnosed_col)
	diagnosed_data = check_unnamed(diagnosed_data)

	print('   ')
	print('Splitting train, test and validation data in ratio 60:25:15')
	# Split train-test-valid in ratio 60:25:15
	split_train = int(diagnosed_data.shape[0] * 0.60)
	split_test = split_train + int(diagnosed_data.shape[0] * 0.25)

	# Convert dataset from pd dataframe to numpy arrays for further processing
	diagnosed_data = np.array(diagnosed_data.astype(float))
	diagnosed_col = np.array(diagnosed_col.astype(float))[:,0]

	# Replace certain labels and then one-hot encode 'diagnosed_for' column
	diagnosed_col = replace_labes(diagnosed_col)
	diagnosed_col = to_categorical(diagnosed_col.astype('int32'), nb_classes=None)

	# Split train, test and validation datasets
	train_data = diagnosed_data[:split_train]
	train_label = diagnosed_col[:split_train]

	test_data = diagnosed_data[split_train:split_test]
	test_label = diagnosed_col[split_train:split_test]

	valid_data = diagnosed_data[split_test:]
	valid_label = diagnosed_col[split_test:]

	print(train_data.shape, train_label.shape)
	print(test_data.shape, test_label.shape)
	print(valid_data.shape, valid_label.shape)
	
	del diagnosed_data, diagnosed_col

	print('Saving Train, Test and Validation data')
	np.save(data_path + '_ALL_diag_train_data.npy', train_data)
	np.save(data_path + '_ALL_diag_test_data.npy', test_data)
	np.save(data_path + '_ALL_diag_valid_data.npy', valid_data)
	np.save(data_path + '_ALL_diag_train_label.npy', train_label)
	np.save(data_path + '_ALL_diag_test_label.npy', test_label)
	np.save(data_path + '_ALL_diag_valid_label.npy', valid_label)

	del train_data, train_label
	del test_data, test_label
	del valid_data, valid_label

def check_if_splitted(force=False):
	file_path_1 = data_path + '_ALL_diag_train_data.npy'
	file_path_2 = data_path + '_ALL_diag_test_data.npy'
	file_path_5 = data_path + '_ALL_diag_valid_data.npy'
	file_path_3 = data_path + '_ALL_diag_train_label.npy'
	file_path_4 = data_path + '_ALL_diag_test_label.npy'
	file_path_6 = data_path + '_ALL_diag_valid_label.npy'
	if ((not os.path.exists(file_path_1)) and (not os.path.exists(file_path_2)) 
		and (not os.path.exists(file_path_3)) and (not os.path.exists(file_path_4))
		and (not os.path.exists(file_path_5)) and (not os.path.exists(file_path_6))):
		split_data()
	else:
		if force:
			split_data()
		else :
			print('Data already splitted')
		
		
check_if_splitted()
