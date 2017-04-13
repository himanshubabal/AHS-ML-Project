import pandas as pd
import numpy as np
import os
from keras.utils.np_utils import to_categorical

def replace_labes(label_data):
	    dict_map = {0.0 : 0.0, 1.0 : 1.0, 2.0 : 2.0, 3.0 : 3.0, 7.0 : 4.0, 9.0 : 5.0,
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

def find_hot_matches(col_index_list_to_remove, hot_col_names_list):
    hot_index_list = list()

    for cold in col_index_list_to_remove:
        for hot in hot_col_names_list:
            if cold in hot:
                index = hot_col_names_list.index(hot)
                if index not in hot_index_list :
                    hot_index_list.append(index)
    return (hot_index_list)

# Pass 'list' of columns to be removed, not individual list
def split_data_by_features(diagnosed_data, diagnosed_col, col_index_list_to_remove):
	# Checking if column 'Unnamed : 0' is in the dataframe, if present, remove it
	diagnosed_col = check_unnamed(diagnosed_col)
	diagnosed_data = check_unnamed(diagnosed_data)

	# List of columns of One-Hot encoded data
	hot_col_names_list = list(diagnosed_data)
	# find matching index of columns of features to be removed
	to_remove_hot_col = find_hot_matches(col_index_list_to_remove, hot_col_names_list)

	# Remove the columns from one-hot encoded data
	diagnosed_data.drop(diagnosed_data.columns[to_remove_hot_col],axis=1,inplace=True,errors='ignore')
	new_hot_col_list = list(diagnosed_data)

	assert (diagnosed_data.shape[0] == diagnosed_col.shape[0])

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

	print('Features Removed : ', col_index_list_to_remove)
	print('Train dataset : ', train_data.shape, train_label.shape)
	print('Test dataset : ', test_data.shape, test_label.shape)
	print('Validation Dataset : ', valid_data.shape, valid_label.shape)

	return(train_data, train_label, test_data, test_label, valid_data, valid_label)
