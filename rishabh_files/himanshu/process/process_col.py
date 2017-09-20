import pandas as pd
import numpy as np
import os
import time
import tensorflow as tf
from keras.utils.np_utils import to_categorical

# Balance binary classes by keeping 50 % data
# of each type, i.e. 0 and 1
# Supply value for the targated class
# i.e. keep BALANCE_THRESHOLD % target class
# in the mixture
BALANCE_THRESHOLD = 50.0

# def replace_labels(label_data, process_0=True):
# 		if 0.0 not in np.unique(label_data):
# 			process_0 = False
# 		dict_map = {0.0 : 0.0, 1.0 : 1.0, 2.0 : 2.0, 3.0 : 3.0, 7.0 : 4.0, 9.0 : 5.0,
# 					19.0 : 6.0, 21.0 : 7.0, 99.0 : 7.0}
# 		cols = list(label_data)
# 		data = np.array(label_data.astype(float))[:,0]
# 		for i in range(len(label_data)):
# 			# element = label_data.iat[i,0]
# 			if data[i] in dict_map:
# 				data[i] = dict_map[float(label_data.iat[i,0])]
# 				# if process_0:
# 				# 	data[i] = dict_map[float(label_data.iat[i,0])]
# 				# else:
# 				# 	data[i] = dict_map[float(label_data.iat[i,0])]-1
# 		df = pd.DataFrame(data, columns=cols)
# 		return df

def replace_labels(label_data):
		# if 0.0 not in np.unique(label_data):
		# 	process_0 = False
		dict_map = {0.0 : 0.0, 1.0 : 1.0, 2.0 : 2.0, 3.0 : 3.0, 7.0 : 4.0, 9.0 : 5.0,
					19.0 : 6.0, 21.0 : 7.0, 99.0 : 7.0}
		cols = list(label_data)
		data = np.array(label_data.astype(float))[:,0]
		for i in range(len(label_data)):
			# element = label_data.iat[i,0]
			if data[i] in dict_map:
				data[i] = dict_map[data[i]]
				# if process_0:
				# 	data[i] = dict_map[float(label_data.iat[i,0])]
				# else:
				# 	data[i] = dict_map[float(label_data.iat[i,0])]-1
		data = data[:,np.newaxis]
		return data

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


def split_data_in_train_test_valid(diagnosed_data, diagnosed_col, to_catg=True):
	assert (diagnosed_data.shape[0] == diagnosed_col.shape[0])

	# Split train-test-valid in ratio 60:25:15
	split_train = int(diagnosed_data.shape[0] * 0.60)
	split_test = split_train + int(diagnosed_data.shape[0] * 0.25)

	# Convert dataset from pd dataframe to numpy arrays for further processing
	diagnosed_data = np.array(diagnosed_data.astype(float))

	# Replace certain labels and then one-hot encode 'diagnosed_for' column
	diagnosed_col = replace_labels(diagnosed_col)
	diagnosed_col = np.array(diagnosed_col.astype(float))[:,0]

	if to_catg:
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
def split_data_by_features(diagnosed_data, diagnosed_col, col_index_list_to_remove, to_catg=True):
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
	

	# Replace certain labels and then one-hot encode 'diagnosed_for' column
	diagnosed_col = replace_labels(diagnosed_col)
	diagnosed_col = np.array(diagnosed_col.astype(float))[:,0]

	if to_catg:
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
		# data_full = pd.concat([diagnosed_data, diagnosed_col], axis=1)

		# assert(len(list(data_full)) == len(list(diagnosed_data)) + len(list(diagnosed_col)))
		# col_name = list(diagnosed_col)[0]
		# assert(col_name in list(data_full))

		# label_data = data_full[data_full[col_name].isin([label])]
		# label_data[col_name] = 1

		# other_size = int(((100.0/BALANCE_THRESHOLD)-1) * label_data.shape[0])
		# other_data = data_full[~data_full[col_name].isin([label])].sample(n=other_size)
		# other_data[col_name] = 0
		
		# final_data = pd.concat([label_data, other_data])
		# # Shuffling the data
		# final_data = final_data.sample(frac=1).reset_index(drop=True)
		
		# column_data = final_data[[col_name]]
		# hot_data = final_data.drop([col_name], inplace=False, axis=1, errors='ignore')

		# assert(hot_data.shape[1] == diagnosed_data.shape[1])
		# assert(column_data.shape[1] == diagnosed_col.shape[1])
		# assert(col_name in list(column_data))

		# return(hot_data, column_data)
		data_np = np.array(diagnosed_data)
		labels_np = np.array(diagnosed_col)

		if len(data_np.shape) == 1:
			data_np = data_np[:,np.newaxis]
		if len(labels_np.shape) == 1:
			labels_np = labels_np[:,np.newaxis]

		full_np = np.concatenate((data_np, labels_np), axis=1)

		np_pos = full_np[full_np[:,-1] == label]
		np_neg = full_np[np.logical_not(full_np[:,-1] == label)]

		np_pos[:,-1] = 1
		np_neg[:,-1] = 0

		other_size = int(((100.0/BALANCE_THRESHOLD)-1) * np_pos.shape[0])
		np_neg_new = np_neg[np.random.choice(np_neg.shape[0], other_size, replace=False)]

		np_new = np.concatenate((np_pos, np_neg_new), axis=0)
		np.random.shuffle(np_new)

		labels_new = np_new[:,-1]
		data_new = np.delete(np.copy(np_new), -1, axis=1)

		if len(labels_new.shape) == 1:
			labels_new = labels_new[:,np.newaxis]

		del np_new, np_pos, np_neg, np_neg_new, full_np
		return(data_new, labels_new)



def split_data_and_make_col_binary(diagnosed_data, diagnosed_col, col_index_list_to_remove, label, balance_classes=True, to_catg=True):
	diagnosed_col = check_unnamed(diagnosed_col)
	diagnosed_data = check_unnamed(diagnosed_data)

	hot_col_names_list = list(diagnosed_data)
	to_remove_hot_col = find_hot_matches(col_index_list_to_remove, hot_col_names_list)

	# Remove the columns from one-hot encoded data
	diagnosed_data.drop(diagnosed_data.columns[to_remove_hot_col],axis=1,inplace=True,errors='ignore')
	new_hot_col_list = list(diagnosed_data)

	assert (diagnosed_data.shape[0] == diagnosed_col.shape[0])

	t1 = time.time()
	# Replace certain labels 
	diagnosed_col = replace_labels(diagnosed_col)
	print('Replace Labels time taken : ', time.time() - t1)

	# diagnosed_col = pd.DataFrame(diagnosed_col)
	# diagnosed_data = pd.DataFrame(diagnosed_data)

	t1 = time.time()
	diagnosed_data, new_col = keep_only_one_label(diagnosed_data, diagnosed_col, label, 
														balance_classes=balance_classes)
	print('Keep one Label time taken : ', time.time() - t1)

	# Split train-test-valid in ratio 70:20:10
	split_train = int(diagnosed_data.shape[0] * 0.70)
	split_test = split_train + int(diagnosed_data.shape[0] * 0.20)

	# Convert dataset from pd dataframe to numpy arrays for further processing
	diagnosed_data = np.array(diagnosed_data.astype(float))
	new_col = np.array(new_col.astype(float))

	# one-hot encode 'diagnosed_for' column
	if to_catg:
		new_col = to_categorical(new_col.astype('int32'))

	final_dict = {}
	hot_dict = {}
	# Split train, test and validation datasets
	hot_dict['train'] = diagnosed_data[:split_train]
	hot_dict['test'] = diagnosed_data[split_train:split_test]
	hot_dict['valid'] = diagnosed_data[split_test:]

	col_dict = {}
	col_dict['train'] = new_col[:split_train]
	col_dict['test'] = new_col[split_train:split_test]
	col_dict['valid'] = new_col[split_test:]

	final_dict['data'] = hot_dict
	final_dict['label'] = col_dict
	
	print('Features Removed : ', col_index_list_to_remove)
	return(final_dict)
