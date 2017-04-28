import pandas as pd
import numpy as np
import os

import tensorflow as tf
from keras.utils.np_utils import to_categorical

# Balance binary classes by keeping 50 % data
# of each type, i.e. 0 and 1
# Supply value for the targated class
# i.e. keep BALANCE_THRESHOLD % target class
# in the mixture
BALANCE_THRESHOLD = 50.0

def replace_labes(label_data, process_0=True):
		if 0.0 not in np.unique(label_data):
			process_0 = False
		dict_map = {0.0 : 0.0, 1.0 : 1.0, 2.0 : 2.0, 3.0 : 3.0, 7.0 : 4.0, 9.0 : 5.0,
					19.0 : 6.0, 21.0 : 7.0, 99.0 : 7.0}
		for i in range(len(label_data)):
			# element = label_data.iloc[i][list(label_data)[0]]
			# element = label_data.iat[i,0]
			if label_data.iat[i,0] in dict_map:
				if process_0:
					label_data.iat[i,0] = dict_map[label_data[i]]
				else:
					label_data.iat[i,0] = dict_map[label_data[i]]-1
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


def split_data_in_train_test_valid(diagnosed_data, diagnosed_col):
	assert (diagnosed_data.shape[0] == diagnosed_col.shape[0])

	# Split train-test-valid in ratio 60:25:15
	split_train = int(diagnosed_data.shape[0] * 0.60)
	split_test = split_train + int(diagnosed_data.shape[0] * 0.25)

	# Convert dataset from pd dataframe to numpy arrays for further processing
	diagnosed_data = np.array(diagnosed_data.astype(float))
	diagnosed_col = np.array(diagnosed_col.astype(float))[:,0]

	# Replace certain labels and then one-hot encode 'diagnosed_for' column
	diagnosed_col = replace_labes(diagnosed_col)
	diagnosed_col = to_categorical(diagnosed_col.astype('int32'))

	# Split train, test and validation datasets
	train_data = diagnosed_data[:split_train]
	train_label = diagnosed_col[:split_train]

	test_data = diagnosed_data[split_train:split_test]
	test_label = diagnosed_col[split_train:split_test]

	valid_data = diagnosed_data[split_test:]
	valid_label = diagnosed_col[split_test:]

	print('Train dataset : ', train_data.shape, train_label.shape)
	print('Test dataset : ', test_data.shape, test_label.shape)
	print('Validation Dataset : ', valid_data.shape, valid_label.shape)

	return(train_data, train_label, test_data, test_label, valid_data, valid_label)


# Pass 'list' of columns to be removed, not individual columns
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
	diagnosed_col = to_categorical(diagnosed_col.astype('int32'))

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


def keep_only_one_label(diagnosed_data, diagnosed_col, label, balance_classes):
	if not balance_classes :
		modified_col = np.copy(diagnosed_col)
		for i in range(len(diagnosed_col)):
			if diagnosed_col[i] == label:
				modified_col[i] = 1
			else :
				modified_col[i] = 0
		return (diagnosed_data, modified_col)
	
	else :		# Balance the classes
		data_full = pd.concat([diagnosed_data, diagnosed_col], axis=1)

		assert(len(list(data_full)) == len(list(diagnosed_data)) + len(list(diagnosed_col)))
		col_name = list(diagnosed_col)[0]
		assert(col_name in list(data_full))

		label_data = data_full[data_full[col_name].isin([label])]
		other_size = int(((100.0/BALANCE_THRESHOLD)-1) * label_data.shape[0])
   		other_data = data_full[~data_full[col_name].isin(label)].sample(n=other_size)
    	
    	final_data = pd.concat([label_data, other_data])
    	# Shuffling the data
    	final_data = final_data.sample(frac=1).reset_index(drop=True)
    	
    	column_data = final_data[[col_name]]
    	hot_data = final_data.drop([col_name], inplace=False, axis=1, errors='ignore')

    	assert(hot_data.shape == diagnosed_data.shape)
    	assert(column_data.shape == diagnosed_col.shape)
    	assert(col_name in list(column_data))

    	return(hot_data, column_data)



def split_data_and_make_col_binary(diagnosed_data, diagnosed_col, col_index_list_to_remove, balance_classes=True):
	diagnosed_col = check_unnamed(diagnosed_col)
	diagnosed_data = check_unnamed(diagnosed_data)

	hot_col_names_list = list(diagnosed_data)
	to_remove_hot_col = find_hot_matches(col_index_list_to_remove, hot_col_names_list)

	# Remove the columns from one-hot encoded data
	diagnosed_data.drop(diagnosed_data.columns[to_remove_hot_col],axis=1,inplace=True,errors='ignore')
	new_hot_col_list = list(diagnosed_data)

	assert (diagnosed_data.shape[0] == diagnosed_col.shape[0])

	# Split train-test-valid in ratio 60:25:15
	split_train = int(diagnosed_data.shape[0] * 0.60)
	split_test = split_train + int(diagnosed_data.shape[0] * 0.25)

	# # Convert dataset from pd dataframe to numpy arrays for further processing
	# diagnosed_data = np.array(diagnosed_data.astype(float))
	# diagnosed_col = np.array(diagnosed_col.astype(float))[:,0]

	# Replace certain labels 
	diagnosed_col = replace_labes(diagnosed_col)

	labels_list = np.unique(diagnosed_col)
	col_dict = {}
	for label in labels_list:
		# diagnosed_col = pd.DataFrame(diagnosed_col)
		# diagnosed_data = pd.DataFrame(diagnosed_data)

		diagnosed_data, new_col = keep_only_one_label(diagnosed_data, diagnosed_col, label, 
															balance_classes=balance_classes)

		# unique, counts = np.unique(new_col, return_counts=True)
		# print ('Unique Values : ', np.asarray((unique, counts)).T)

		# Convert dataset from pd dataframe to numpy arrays for further processing
		diagnosed_data = np.array(diagnosed_data.astype(float))
		new_col = np.array(new_col.astype(float))[:,0]

		# one-hot encode 'diagnosed_for' column
		new_col = to_categorical(new_col.astype('int32')) 
		data_dict = {}
		data_dict['train'] = new_col[:split_train]
		data_dict['test'] = new_col[split_train:split_test]
		data_dict['valid'] = new_col[split_test:]

		col_dict[label] = data_dict
	

	# Split train, test and validation datasets
	train_data = diagnosed_data[:split_train]
	test_data = diagnosed_data[split_train:split_test]
	valid_data = diagnosed_data[split_test:]

	print('Features Removed : ', col_index_list_to_remove)
	return(train_data, test_data, valid_data, col_dict, labels_list)
