import time
import argparse
import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd

# Import from sibling directory
from ..process.process_col import split_data_and_make_col_binary, split_data_in_train_test_valid, check_unnamed
from ..models.FCNN import FFNN
from ..models.FCNN_graph import TF_GRAPH
# Import from parent directory
from ..variables import data_path, model_save_path, data_scratch, BATCH_SIZE
from ..train import TRAIN_NN
from ..evaluation import *

# data_path = data_scratch
activations = [tf.nn.relu]
keep_prob = 0.7

parser = argparse.ArgumentParser(description='Pass LABLE in -label which is to be removed')
parser.add_argument('-label', default = 1,  type=int, help = 'State for which data is to be processed -- 5, 8, 9, 10, 18, 20, 21, 22, 23')
parser.add_argument('-state', default=22, type=int, help = 'State for which data ' + 
								'is to be processed -- 5, 8, 9, 10, 18, 20, 21, 22, 23')
parser.add_argument('-dtype', default='COMB', type=str, help = 'Type of Dataset - ' +
														  'COMB, MORT, WOMEN, WPS, CAB')
parser.add_argument('-col', default='diagnosed_for', type=str, help = 'Column to ' +
		  'be predicted -- diagnosed_for, illness_type, symptoms_pertaining_to_illness')
# parser.add_argument('-include_0', default=False, type=bool, help='To include 0(false ' +
# 											  ' cases) in to-be-predicted column or not')
parser.add_argument('-epochs', default=1001, type=int, help='No of Epochs')
parser.add_argument('--include_0', action='store_true', default=False, help='To include 0(false ' +
											  ' cases) in to-be-predicted column or not')
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
EPOCHS = int(args.epochs)

if label_to_be_kept not in labels_list :
	print('Please enter -label from [0.,  1.,  2.,  3.,  4.,  5.,  6.]')
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

print('LABEL : ', label_to_be_kept, 'state : ', state, '  Include 0 : ', INCLUDE_0)

# Default file names
dataset_save_path = str(int(state)) + '/' + str(state) + '_AHS_' + dtype

if not INCLUDE_0 :
	diag_col_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_col' + '.csv'
	diag_hot_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_hotData' + '.csv'
else :
	diag_col_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_col' + '_with_0' + '.csv'
	diag_hot_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_hotData' + '_with_0' + '.csv'

if INCLUDE_0:
	txt = 'including_0'
else :
	txt = 'excluding_0'


# BATCH_SIZE = 2048
FINAL_RES = {}
print('EPOCHS : ', EPOCHS, '  BATCH_SIZE : ', BATCH_SIZE)

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

	if (cold_col_names_list[k] == 'diagnosed_for'):
		final_dict = split_data_and_make_col_binary(diagnosed_data.copy(), diagnosed_col.copy(), [], label_to_be_kept)
	else :
		final_dict = split_data_and_make_col_binary(diagnosed_data.copy(), diagnosed_col.copy(), [cold_col_names_list[k]], label_to_be_kept)
	
	label_list = [0,1]
	label = label_to_be_kept

	train_data = final_dict['data']['train']
	test_data = final_dict['data']['test']
	valid_data = final_dict['data']['valid']
	
	train_label = final_dict['label']['train']
	test_label = final_dict['label']['test']
	valid_label = final_dict['label']['valid']

	print('Train : ', train_data.shape, train_label.shape)
	print('Test : ', test_data.shape, test_label.shape)
	print('Valid : ', valid_data.shape, valid_label.shape)
	
	
	model_path = model_save_path + str(pred_col) + '/' + txt + '/' + str(state) + '/' + \
		str(label_to_be_kept) + '/' +str(cold_col_names_list[k]) + '/' + '8_hidden_layers' + '/'
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	if (BATCH_SIZE*10 > train_label.shape[0]):
		BATCH_SIZE = int(train_label.shape[0]/15)
		print('New Batch Size : ', BATCH_SIZE)

	graph_nn, param_array = TF_GRAPH(data_shape=train_data.shape[1], label_shape=train_label.shape[1], BATCH_SIZE=BATCH_SIZE,
						learning_rate=0.001).create_graph()

	train_nn_object = TRAIN_NN(tf_graph=graph_nn, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS, param_array=param_array, 
		label_list=label_list, train_data=train_data, train_label=train_label, model_path=model_path, 
		model_name='8_hidden_layers', keep_prob=keep_prob, test_data=test_data, test_label=test_label, 
		valid_data=np.copy(test_data), valid_label=np.copy(test_label), save_model=True, load_model_if_saved=False)

	FINAL_RES[cold_col_names_list[k]] = train_nn_object.train()['valid_error']
	print('Time Taken - full : ' + str((time.time() - t1)/60) + ' minutes')

print('')
print('-------------------------')
print('-------------------------')
print(FINAL_RES)

del diagnosed_data, diagnosed_col
sys.modules[__name__].__dict__.clear()