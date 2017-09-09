# import tensorflow as tf

# import pandas as pd
# import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# import math
# import os
# import argparse
# import pickle
# import operator

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD
# from keras.utils.np_utils import to_categorical

# from python_helper.sort_clean_data import create_balanced_classes

# data_path = '/home/physics/btech/ph1140797/AHS-ML-Project/data/'
# # data_path = '/home/physics/btech/ph1140797/scratch/AHS_data/'
# clean_csv_name = '22_AHS_COMB_Clean.csv'
# NaN_TOLRENCE = 100.0			# [30, 50, 70, 100]
# print('------------------------------')
# print('NAN Tolerence : ', NaN_TOLRENCE)
# print('------------------------------')

# def one_hot_df(data_frame, one_hot_colnames=list()) :
# 	if len(one_hot_colnames) != 0:
# 		colnames = list(data_frame)
# 		hot_col = list()
# 		for hot in one_hot_colnames :
# 			if hot in colnames :
# 				hot_col.append(hot)
# 	else:
# 		hot_col = list(data_frame)
# 	if 'district' in hot_col :
# 		hot_col.remove('district')
# 	if 'state' in hot_col :
# 		hot_col.remove('state')
# 	if 'age' in hot_col:
# 		hot_col.remove('age')     
# 	data_frame = pd.get_dummies(data_frame, columns=hot_col)
# 	return (data_frame)


# col_to_be_removed = [
# 	'v104'
# 	'Unnamed: 0',
# 	'psu_id',
# 	'house_no',
# 	'house_hold_no',
# 	'member_identity',
# 	'father_serial_no',
# 	'mother_serial_no',
# 	'date_of_birth',
# 	'month_of_birth',
# 	'year_of_birth',
# 	'date_of_marriage',
# 	'month_of_marriage',
# 	'year_of_marriage',
# 	'building_no',
# 	'no_of_dwelling_rooms',
# 	'rural_1',
# 	'rural_2',
# 	'stratum_code',
# 	'relation_to_head',
# 	'member_identity',
# 	'father_serial_no',
# 	'mother_serial_no',
# 	'date_of_birth',
# 	'month_of_birth',
# 	'year_of_birth',
# 	'date_of_marriage',
# 	'month_of_marriage',
# 	'year_of_marriage',
# 	'isheadchanged',
# 	'year'
# ]

# def missing_values_table(df): 
# 	mis_val = 100 * df.isnull().sum()/len(df)
# 	mis_val_index = mis_val.index.tolist()
# 	mis_val_val = mis_val.tolist()
# 	val_dict = {}
# 	for i in range(len(mis_val_index)):
# 		val_dict[mis_val_index[i]] = mis_val_val[i]
# 	# Sort it by values
# 	val_dict = sorted(val_dict.items(), key=operator.itemgetter(1), reverse=True)
# 	return val_dict 

# def prep_for_analysis(drop_col_with_almost_nan=True):
# 	dist = pd.read_csv(data_path + clean_csv_name)
# 	print('Clean Data shape : ', dist.shape)

# 	print('  ')
# 	print('Removing not-so-useful columns')
# 	# Dropping the above columns
# 	dist = dist.drop(col_to_be_removed, axis=1, errors='ignore')
# 	print('Shape after removing : ', dist.shape)


# 	if 'v104' in list(dist):
# 		dist.drop(['v104'], inplace=True, axis=1, errors='ignore')
# 	assert ('v104' not in list(dist))

# 	dist_p = dist.dropna(subset=['diagnosed_for'])
# 	print('Drop NAN shape : ', dist_p.shape)
# 	del dist

# 	dist_p = create_balanced_classes(dist_p, [1.0,2.0,3.0,7.0,9.0,19.0,21.0,99.0],"diagnosed_for")
# 	print('After balance : ', dist_p.shape)


# 	if drop_col_with_almost_nan:
# 		nan_dict = missing_values_table(dist_p)
# 		nan_col = list()

# 		for col in nan_dict:
# 			if col[1] > NaN_TOLRENCE :
# 				nan_col.append(col[0])
# 		print('Nan Col : ', nan_col)

# 		dist_p = dist_p.drop(nan_col, axis=1, errors='ignore')
# 		print('After Dropping Nan Col Shape : ', dist_p.shape)

# 	# Shuffling the dataset and reset index
# 	dist_p = dist_p.iloc[np.random.permutation(len(dist_p))]
# 	dist_p = dist_p.reset_index(drop=True)

# 	print('Splitting to-predict column')
# 	# Seperating 'diagnosed_for' variable for prediction
# 	diagnosed_col = dist_p[['diagnosed_for']]
# 	diagnosed_data = dist_p.drop(['diagnosed_for'], inplace=False, axis=1, errors='ignore')
# 	# pd.DataFrame(list(dist_p)).to_csv(data_path + 'diagnosed_for_col_names_nan.csv', index=False)
# 	del dist_p

# 	print('One Hot Encoding Data')
# 	try:
# 		diagnosed_data = one_hot_df(diagnosed_data)
# 	except Exception as e:
# 		size_threshold = 30000
# 		no_of_df = int(diagnosed_data.shape[0]/size_threshold)
# 		splitted_dataset = np.array_split(diagnosed_data, no_of_df)
# 		df_list = list()
# 		for df in splitted_dataset:
# 			df_list.append(pd.DataFrame(df))

# 		hot_df_list = list()
# 		for df in df_list:
# 			hot_df_list.append(one_hot_df(df))

# 		diagnosed_data = pd.concat(hot_df_list)

# 	diagnosed_data = diagnosed_data.fillna(0)
# 	print('Hot Encoded shape : ', diagnosed_data.shape)
# 	return (diagnosed_data, diagnosed_col)


# def replace_labes(label_data):
# 		dict_map = {0.0 : 0.0, 1.0 : 1.0, 2.0 : 2.0, 3.0 : 3.0, 7.0 : 4.0, 9.0 : 5.0,
# 					19.0 : 6.0, 21.0 : 7.0, 99.0 : 7.0}
# 		for i in range(len(label_data)):
# 			if label_data[i] in dict_map:
# 				label_data[i] = dict_map[label_data[i]]-1
# 			else :
# 				label_data[i] = 0.0
# 		return label_data

# def check_unnamed(dataframe):
# 	if 'Unnamed: 0' in list(dataframe):
# 		dataframe = dataframe.drop('Unnamed: 0',axis=1,errors='ignore')
# 	return dataframe


# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# diagnosed_data, diagnosed_col = prep_for_analysis()

# diagnosed_col = check_unnamed(diagnosed_col)
# diagnosed_data = check_unnamed(diagnosed_data)

# assert (diagnosed_data.shape[0] == diagnosed_col.shape[0])
# split_index = int(diagnosed_data.shape[0] * 0.75)

# print('   ')
# print('Splitting train and test data in ratio 75:25')
# train_data = np.array(diagnosed_data.astype(float))[:split_index]
# train_label = np.array(diagnosed_col.astype(float))[:split_index][:,0]

# test_data = np.array(diagnosed_data.astype(float))[split_index:]
# test_label = np.array(diagnosed_col.astype(float))[split_index:][:,0]

# train_rep = replace_labes(train_label)
# test_rep = replace_labes(test_label)

# train_label = to_categorical(train_rep.astype('int32'), nb_classes=None)
# test_label = to_categorical(test_rep.astype('int32'), nb_classes=None)

# print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
# del diagnosed_data, diagnosed_col

# np.save(data_path + '22_train_data.npy', train_data)
# np.save(data_path + '22_test_data.npy', test_data)
# np.save(data_path + '22_train_label.npy', train_label)
# np.save(data_path + '22_test_label.npy', test_label)
