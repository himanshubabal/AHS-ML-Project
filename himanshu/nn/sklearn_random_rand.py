import time
import argparse
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pandas as pd

# Import from sibling directory
from ..process.process_col import split_data_and_make_col_binary, split_data_in_train_test_valid, check_unnamed, keep_only_one_label
# from ..process.process_all import one_hot_df
from ..models.sklearn_classifiers import SKLearn_models
# Import from parent directory
from ..variables import data_path, model_save_path, data_scratch


parser = argparse.ArgumentParser(description='Pass LABLE in -label which is to be removed')
parser.add_argument('-state', default=22, type=int, help = 'State for which data ' + 
								'is to be processed -- 5, 8, 9, 10, 18, 20, 21, 22, 23')
parser.add_argument('-label', default = 1,  type=int, help = 'label which is to be removed' +
													' -- [1., 2., 3., 4., 5., 6., 7.]')
parser.add_argument('-dtype', default='COMB', type=str, help = 'Type of Dataset - ' +
														  'COMB, MORT, WOMEN, WPS, CAB')
parser.add_argument('-col', default='diagnosed_for', type=str, help = 'Column to ' +
		  'be predicted -- diagnosed_for, illness_type, symptoms_pertaining_to_illness')
parser.add_argument('--include_0', action='store_true', default=False, help='To include 0(false ' +
											  ' cases) in to-be-predicted column or not')
parser.add_argument('--rand_forest', action='store_true', default=False, help='True for Random Forese' + 
												' False for Logistic Regression ')

args = parser.parse_args()


states_list = [5, 8, 9, 10, 18, 20, 21, 22, 23]
dtypes_list = ['COMB', 'MORT', 'WOMEN', 'WPS', 'CAB']
pred_col_list = ['diagnosed_for', 'illness_type', 'symptoms_pertaining_to_illness']
bool_list = [True, False]
labels_list = [0.,  1.,  2.,  3.,  4.,  5.,  6., 7.]

label_to_be_kept = float(args.label)
state = int(args.state)
dtype = str(args.dtype)
pred_col = str(args.col)
INCLUDE_0 = args.include_0
IS_RAND_FOREST = args.rand_forest


if label_to_be_kept not in labels_list :
	print('Please enter -label from [1.,  2.,  3.,  4.,  5.,  6., 7.]')
	sys.exit()
if state not in states_list :
	print('Please enter -state from [5, 8, 9, 10, 18, 20, 21, 22, 23]')
	sys.exit()
if dtype not in dtypes_list :
	print('Please enter -dtype from [COMB, MORT, WOMEN, WPS, CAB]')
	sys.exit()
if pred_col not in pred_col_list :
	print('Please enter -col from [diagnosed_for, illness_type, ' +
									'symptoms_pertaining_to_illness]')
	sys.exit()


if IS_RAND_FOREST:
	print('Using Random Forest Classifier')
else:
	print('Using Logistic Regression Classifier')

# Random Forest Trained model takes too much disk space
# Save model only if Logistic Regression is used
# not if random forest is used
to_save = True
if IS_RAND_FOREST :
	to_save = False

# Default file names
dataset_save_path = str(int(state)) + '/' + str(state) + '_AHS_' + dtype
clean_data_path = dataset_save_path + '_' + 'Clean.csv'

if not INCLUDE_0 :
	diag_col_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_col' + '.csv'
	diag_hot_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_hotData' + '.csv'
	diag_cold_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_coldData' + '.csv'
else :
	diag_col_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_col' + '_with_0' + '.csv'
	diag_hot_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_hotData' + '_with_0' + '.csv'
	diag_cold_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_coldData' + '_with_0' + '.csv'

print('Dataset Used : ', diag_hot_csv_name)
print('Labels Used : ', diag_col_csv_name)

FINAL_RES = {}


cold_data = pd.read_csv(data_path + diag_cold_csv_name)
to_pred_col = pd.read_csv(data_path + diag_col_csv_name)

# Dropping column 'state'
cold_data = cold_data.drop(['state'], axis=1, errors='ignore')

# List of column names
cold_col_names_list = list(cold_data)


def replace_col_by_random_values(dataframe, col_to_be_replaced_list):
	df_length = dataframe.shape[0]

	for col in col_to_be_replaced_list:
		unique_val = np.unique(dataframe[[col]])
		col_index = dataframe.columns.get_loc(col)
		dataframe = dataframe.drop(col, axis=1)

		rand_array = np.random.choice(unique_val, df_length)
		dataframe.insert(col_index, col, rand_array)

	return dataframe

def one_hot_df(data_frame, one_hot_colnames=list()) :
	if len(one_hot_colnames) != 0:
		colnames = list(data_frame)
		hot_col = list()
		for hot in one_hot_colnames :
			if hot in colnames :
				hot_col.append(hot)
	else:
		hot_col = list(data_frame)

	# Remove district, age and state from col list
	if 'district' in hot_col :
		hot_col.remove('district')
	if 'state' in hot_col :
		hot_col.remove('state')
	if 'age' in hot_col:
		hot_col.remove('age')

	# get_dummies() one hot encodes the dataframe
	data_frame = pd.get_dummies(data_frame, columns=hot_col)
	return (data_frame)

t1 = time.time()

diagnosed_d = cold_data
diagnosed_c = to_pred_col

rand_col = np.random.choice(list(diagnosed_d), 0)#len(list(diagnosed_d))//6)
print(rand_col.shape)
# print(set(list(diagnosed_d)) - set(rand_col))
# rand_col = ['cart', 'house_structure', 'highest_qualification', 'toilet_used']

diagnosed_d = replace_col_by_random_values(diagnosed_d, rand_col)

# print(diagnosed_d.shape)
diagnosed_d = one_hot_df(diagnosed_d)
# print(diagnosed_d.shape)
# print(diagnosed_d.head())

final_dict = split_data_and_make_col_binary(diagnosed_d, diagnosed_c, 
				[], label_to_be_kept, to_catg=False)

train_data = final_dict['data']['train']
test_data = final_dict['data']['test']
valid_data = final_dict['data']['valid']

train_label = final_dict['label']['train']
test_label = final_dict['label']['test']
valid_label = final_dict['label']['valid']

print('Train : ', train_data.shape, train_label.shape)
print('Test : ', test_data.shape, test_label.shape)
print('Valid : ', valid_data.shape, valid_label.shape)

if INCLUDE_0:
	txt = 'including_0'
else :
	txt = 'excluding_0'

if IS_RAND_FOREST:
	model_name = 'random_forest'
else:
	model_name = 'logistic_regression'

model_path = model_save_path

log_reg = SKLearn_models(train_data, train_label, test_data, test_label, valid_data, 
	valid_label, model_path, model_name, save_model=False, rand_forest=IS_RAND_FOREST)

res_dict = log_reg.classify()
print('Time Taken - full : ' + str((time.time() - t1)/60) + ' minutes')












del cold_data, to_pred_col
sys.modules[__name__].__dict__.clear()









