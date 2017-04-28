import time
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import argparse

# Import from sibling directory
# from ..process.process import *
from ..process.process_col import split_data_by_features, split_data_in_train_test_valid, check_unnamed
from ..models.FCNN import FFNN
from ..models.FCNN_graph import TF_GRAPH
# Import from parent directory
from ..variables import data_path, model_save_path, data_scratch
from ..train import TRAIN_NN
from ..evaluation import *


parser = argparse.ArgumentParser(description='-state, -dtype, -col, -include_0')
parser.add_argument('-state', default=22, type=int, help = 'State for which data ' + 
								'is to be processed -- 5, 8, 9, 10, 18, 20, 21, 22, 23')
parser.add_argument('-dtype', default='COMB', type=str, help = 'Type of Dataset - ' +
														  'COMB, MORT, WOMEN, WPS, CAB')
parser.add_argument('-col', default='diagnosed_for', type=str, help = 'Column to ' +
		  'be predicted -- diagnosed_for, illness_type, symptoms_pertaining_to_illness')
parser.add_argument('-include_0', default=False, type=bool, help='To include 0(false ' +
											  ' cases) in to-be-predicted column or not')
args = parser.parse_args()

states_list = [5, 8, 9, 10, 18, 20, 21, 22, 23]
dtypes_list = ['COMB', 'MORT', 'WOMEN', 'WPS', 'CAB']
pred_col_list = ['diagnosed_for', 'illness_type', 'symptoms_pertaining_to_illness']
bool_list = ['True', 'False']

state = int(args.state)
dtype = str(args.dtype)
pred_col = str(args.col)
INCLUDE_0 = str(args.include_0)

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
if INCLUDE_0 not in bool_list:
	print('Please enter -remove_nan and -include_0 as True or False')
	sys.exit()

# Convert String to Bool
def _to_bool(str):
	if str == 'True':
		return True
	else :
		return False

INCLUDE_0 = _to_bool(INCLUDE_0)

# Default file names
dataset_save_path = str(int(state)) + '/' + str(state) + '_AHS_' + dtype

if not INCLUDE_0 :
	diag_col_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_col' + '.csv'
	diag_hot_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_hotData' + '.csv'
else :
	diag_col_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_col' + '_with_0' + '.csv'
	diag_hot_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_hotData' + '_with_0' + '.csv'


keep_prob = 0.7
label_list = [0,1,2,3,4,5,6]

BATCH_SIZE = 1024
EPOCHS = 101
FINAL_RES = {}
print('EPOCHS : ', EPOCHS, '  BATCH_SIZE : ', BATCH_SIZE)

data_path = data_scratch

diagnosed_data = pd.read_csv(data_path + diag_hot_csv_name)
diagnosed_col = pd.read_csv(data_path + diag_col_csv_name)

train_data, train_label, test_data, test_label, valid_data, valid_label = split_data_in_train_test_valid(diagnosed_data.copy(), diagnosed_col.copy())

model_path = model_save_path + 'full_data' + '/'
if not os.path.exists(model_path):
	os.makedirs(model_path)

graph_nn, param_array = TF_GRAPH(data_shape=train_data.shape[1], label_shape=train_label.shape[1], BATCH_SIZE=BATCH_SIZE,
						learning_rate=0.001).create_graph()

print('')
print('Working with full dataset initially')

train_nn_object = TRAIN_NN(tf_graph=graph_nn, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS, param_array=param_array, 
	label_list=label_list, train_data=train_data, train_label=train_label, model_path=model_path, 
	model_name='8_hidden_layers', keep_prob=keep_prob, test_data=test_data, test_label=test_label, 
	valid_data=valid_data, valid_label=valid_label, save_model=False, load_model_if_saved=False)

valid_error = train_nn_object.train()
print('Full Dataset Error : ', valid_error)

del train_data, train_label, test_data, test_label, valid_data, valid_label



print('')
print('')
print('Now testing by removing one feature each time')
cold_col_names_list = pd.read_csv(data_path + dataset_save_path +'_'+ pred_col +'_' + 'col_names.csv')
cold_col_names_list = check_unnamed(cold_col_names_list)
cold_col_names_list = cold_col_names_list[cold_col_names_list.columns[0]].tolist()

for k in range(len(cold_col_names_list)):
	t1 = time.time()
	print('-------------------------------------------------------')
	print("current feature to be ablated : ",cold_col_names_list[k])

	train_data, train_label, test_data, test_label, valid_data, valid_label = split_data_by_features(diagnosed_data.copy(), diagnosed_col.copy(), [cold_col_names_list[k]])

	model_path = model_save_path + str('all_labels') + '/' + str(cold_col_names_list[k]) + '/'
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	# param_array = (_optimize, _error, y_pred, y_true, model_saver, X, y, keep_prob)
	graph_nn, param_array = TF_GRAPH(data_shape=train_data.shape[1], label_shape=train_label.shape[1], BATCH_SIZE=BATCH_SIZE,
						learning_rate=0.001).create_graph()

	train_nn_object = TRAIN_NN(tf_graph=graph_nn, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS, param_array=param_array, 
		label_list=label_list, train_data=train_data, train_label=train_label, model_path=model_path, 
		model_name='8_hidden_layers', keep_prob=keep_prob, test_data=test_data, test_label=test_label, 
		valid_data=np.copy(test_data), valid_label=np.copy(test_label), save_model=False, load_model_if_saved=False)

	FINAL_RES[cold_col_names_list[k]] = train_nn_object.train()['valid_error']


print('')
print('-------------------------')
print('-------------------------')
print(FINAL_RES)
