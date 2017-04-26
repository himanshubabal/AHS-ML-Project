import time
import numpy as np
import os
import tensorflow as tf
import pandas as pd

# Import from sibling directory
from ..process.process import *
from ..process.process_col import split_data_by_features
from ..models.FCNN import FFNN
from ..models.FCNN_graph import TF_GRAPH
# Import from parent directory
from ..variables import data_path, model_save_path
from ..train import TRAIN_NN
from ..evaluation import *

keep_prob = 0.7
label_list = [0,1,2,3,4,5,6]

BATCH_SIZE = 1024
EPOCHS = 1
FINAL_RES = {}
print('EPOCHS : ', EPOCHS, '  BATCH_SIZE : ', BATCH_SIZE)



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
	valid_data=np.copy(test_data), valid_label=np.copy(test_label), save_model=False, load_model_if_saved=False)

valid_error = train_nn_object.train()
print('Full Dataset Error : ', valid_error)

del train_data, train_label, test_data, test_label



print('')
print('')
print('Now testing by removing one feature each time')
cold_col_names_list = pd.read_csv(data_path + 'diagnosed_for_col_names.csv', low_memory=False)
cold_col_names_list = check_unnamed(cold_col_names_list)
cold_col_names_list = cold_col_names_list[cold_col_names_list.columns[0]].tolist()

for k in range(len(cold_col_names_list)):
	t1 = time.time()
	diagnosed_data = pd.read_csv(data_path + '22_COMB_diag_hotData.csv', low_memory=False)
	diagnosed_col = pd.read_csv(data_path + '22_COMB_diag_col.csv', low_memory=False)

	print('-------------------------------------------------------')
	print("current feature to be ablated : ",cold_col_names_list[k])

	train_data, train_label, test_data, test_label, valid_data, valid_label = split_data_by_features(diagnosed_data, diagnosed_col, [cold_col_names_list[k]])

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
