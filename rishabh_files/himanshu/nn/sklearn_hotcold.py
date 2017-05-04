import time
import argparse
import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd

# Import from sibling directory
from ..process.process_col import split_data_by_features, split_data_in_train_test_valid, check_unnamed
from ..models.sklearn_classifiers import SKLearn_models
# Import from parent directory
from ..variables import data_path, model_save_path, data_scratch

# data_path = data_scratch

parser = argparse.ArgumentParser(description='Pass required args')
parser.add_argument('-state', default=22, type=int, help = 'State for which data ' + 
								'is to be processed -- 5, 8, 9, 10, 18, 20, 21, 22, 23')
parser.add_argument('-dtype', default='COMB', type=str, help = 'Type of Dataset - ' +
														  'COMB, MORT, WOMEN, WPS, CAB')
parser.add_argument('-col', default='diagnosed_for', type=str, help = 'Column to ' +
		  'be predicted -- diagnosed_for, illness_type, symptoms_pertaining_to_illness')
# parser.add_argument('-include_0', default=False, type=bool, help='To include 0(false ' +
# 											  ' cases) in to-be-predicted column or not')
# parser.add_argument('-rand_forest', default=False, type=bool, help='True for Random Forese' + 
												# ' False for Logistic Regression ')
parser.add_argument('--include_0', action='store_true', default=False, help='To include 0(false ' +
											  ' cases) in to-be-predicted column or not')
parser.add_argument('--rand_forest', action='store_true', default=False, help='True for Random Forese' + 
												' False for Logistic Regression ')
args = parser.parse_args()

states_list = [5, 8, 9, 10, 18, 20, 21, 22, 23]
dtypes_list = ['COMB', 'MORT', 'WOMEN', 'WPS', 'CAB']
pred_col_list = ['diagnosed_for', 'illness_type', 'symptoms_pertaining_to_illness']
bool_list = [True, False]

state = int(args.state)
dtype = str(args.dtype)
pred_col = str(args.col)
INCLUDE_0 = args.include_0
IS_RAND_FOREST = args.rand_forest

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

if INCLUDE_0:
	txt = 'including_0'
else :
	txt = 'excluding_0'

if IS_RAND_FOREST:
	model_name = 'random_forest'
else:
	model_name = 'logistic_regression'

# Random Forest Trained model takes too much disk space
# Save model only if Logistic Regression is used
# not if random forest is used
to_save = True
if IS_RAND_FOREST :
	to_save = False

# Default file names
dataset_save_path = str(int(state)) + '/' + str(state) + '_AHS_' + dtype

if not INCLUDE_0 :
	diag_col_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_col' + '.csv'
	diag_hot_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_hotData' + '.csv'
else :
	diag_col_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_col' + '_with_0' + '.csv'
	diag_hot_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_hotData' + '_with_0' + '.csv'

print('Dataset Used : ', diag_hot_csv_name)
print('Labels Used : ', diag_col_csv_name)

print('Initially doing on Full Dataset')

diagnosed_data = pd.read_csv(data_path + diag_hot_csv_name)
diagnosed_col = pd.read_csv(data_path + diag_col_csv_name)


train_data, train_label, test_data, test_label, valid_data, valid_label = \
		split_data_in_train_test_valid(diagnosed_data.copy(), diagnosed_col.copy(), to_catg=False)

model_path = model_save_path + str(pred_col) + '/' + txt + '/' + str(state) + '/' + \
												'full_data' + '/' + 'sklearn' + '/'
if not os.path.exists(model_path):
	os.makedirs(model_path)

log_reg = SKLearn_models(train_data, train_label, test_data, test_label, valid_data, 
	valid_label, model_path, model_name, save_model=to_save, rand_forest=IS_RAND_FOREST)

log_reg.classify()

del train_data, train_label, test_data, test_label, valid_data, valid_label



FINAL_RES = {}

print('')
print('')
print('Testing by removing one feature each time')
cold_col_names_list = pd.read_csv(data_path + dataset_save_path +'_'+ pred_col +'_' + 'col_names.csv')
cold_col_names_list = check_unnamed(cold_col_names_list)
cold_col_names_list = cold_col_names_list[cold_col_names_list.columns[0]].tolist()

diagnosed_data = pd.read_csv(data_path + diag_hot_csv_name)
diagnosed_col = pd.read_csv(data_path + diag_col_csv_name)

for k in range(len(cold_col_names_list)):
# for k in range(1):
	t1 = time.time()

	print('-------------------------------------------------------')
	print("current feature to be ablated : ",cold_col_names_list[k])

	print('Original : ', diagnosed_data.shape, diagnosed_col.shape)

	train_data, train_label, test_data, test_label, valid_data, valid_label = split_data_by_features(diagnosed_data.copy(), 
		diagnosed_col.copy(), [cold_col_names_list[k]], to_catg=False)
	
	model_path = model_save_path + str(pred_col) + '/' + txt + '/' + str(state) + '/' + \
		'all_labels_feature_removal' + '/' +str(cold_col_names_list[k]) + '/' + 'sklearn' + '/'
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	log_reg = SKLearn_models(train_data, train_label, test_data, test_label, valid_data, 
		valid_label, model_path, model_name, save_model=to_save, rand_forest=IS_RAND_FOREST)

	FINAL_RES[cold_col_names_list[k]] = log_reg.classify()['error']
	print('Time Taken - full : ' + str((time.time() - t1)/60) + ' minutes')

print('')
print('-------------------------')
print('-------------------------')
print(FINAL_RES)

del diagnosed_data, diagnosed_col
sys.modules[__name__].__dict__.clear()
