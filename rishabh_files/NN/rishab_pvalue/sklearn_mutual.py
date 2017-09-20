import time
import argparse
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pandas as pd

from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.linear_model import LogisticRegression

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

hot_data = pd.read_csv(data_path + diag_hot_csv_name)
to_pred_col = pd.read_csv(data_path + diag_col_csv_name)

# Dropping column 'state'
hot_data = hot_data.drop(['state'], axis=1, errors='ignore')


hot_columns = list(hot_data.columns.values)
# data = data.as_matrix()
# data_Y = np.reshape(data_Y.as_matrix(),(-1,))
# print data_Y.shape
classif = mutual_info_classif(hot_data, to_pred_col)
a = classif
f = [hot_columns[i] for i in range(len(a)) if a[i]==0.0]
hot_data.drop(f, axis = 1, inplace = True)

train_data, train_label, test_data, test_label, valid_data, valid_label = \
		split_data_in_train_test_valid(hot_data, to_pred_col, to_catg=False)

model_path = model_save_path + str(pred_col) + '/' + txt + '/' + str(state) + '/' + \
												'full_data' + '/' + 'sklearn' + '/'
if not os.path.exists(model_path):
	os.makedirs(model_path)

log_reg = SKLearn_models(train_data, train_label, test_data, test_label, valid_data, 
	valid_label, model_path, model_name, save_model=to_save, rand_forest=IS_RAND_FOREST)

log_reg.classify()






