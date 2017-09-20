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
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, IsolationForest
data_path = '/home/physics/btech/ph1140797/AHS-ML-Project/data/'
model_save_path = '/home/physics/btech/ph1140797/AHS-ML-Project/saved_models/'

# Import from sibling directory
# from ..process.process_col import split_data_and_make_col_binary, split_data_in_train_test_valid, check_unnamed, keep_only_one_label
# from ..process.process_all import one_hot_df
# from ..models.sklearn_classifiers import SKLearn_models
# Import from parent directory
# from ..variables import data_path, model_save_path, data_scratch

# parser = argparse.ArgumentParser(description='Pass LABLE in -label which is to be removed')
# parser.add_argument('-state', default=22, type=int, help = 'State for which data ' + 
# 								'is to be processed -- 5, 8, 9, 10, 18, 20, 21, 22, 23')
# parser.add_argument('-label', default = 1,  type=int, help = 'label which is to be removed' +
# 													' -- [1., 2., 3., 4., 5., 6., 7.]')
# parser.add_argument('-dtype', default='COMB', type=str, help = 'Type of Dataset - ' +
# 														  'COMB, MORT, WOMEN, WPS, CAB')
# parser.add_argument('-col', default='diagnosed_for', type=str, help = 'Column to ' +
# 		  'be predicted -- diagnosed_for, illness_type, symptoms_pertaining_to_illness')
# parser.add_argument('--include_0', action='store_true', default=False, help='To include 0(false ' +
# 											  ' cases) in to-be-predicted column or not')
# parser.add_argument('--rand_forest', action='store_true', default=False, help='True for Random Forese' + 
# 												' False for Logistic Regression ')

# args = parser.parse_args()


states_list = [5, 8, 9, 10, 18, 20, 21, 22, 23]
dtypes_list = ['COMB', 'MORT', 'WOMEN', 'WPS', 'CAB']
pred_col_list = ['diagnosed_for', 'illness_type', 'symptoms_pertaining_to_illness']
bool_list = [True, False]
labels_list = [0.,  1.,  2.,  3.,  4.,  5.,  6., 7.]

# label_to_be_kept = float(args.label)
# state = int(args.state)
# dtype = str(args.dtype)
# pred_col = str(args.col)
# INCLUDE_0 = args.include_0
# IS_RAND_FOREST = args.rand_forest


# if label_to_be_kept not in labels_list :
# 	print('Please enter -label from [1.,  2.,  3.,  4.,  5.,  6., 7.]')
# 	sys.exit()
# if state not in states_list :
# 	print('Please enter -state from [5, 8, 9, 10, 18, 20, 21, 22, 23]')
# 	sys.exit()
# if dtype not in dtypes_list :
# 	print('Please enter -dtype from [COMB, MORT, WOMEN, WPS, CAB]')
# 	sys.exit()
# if pred_col not in pred_col_list :
# 	print('Please enter -col from [diagnosed_for, illness_type, ' +
# 									'symptoms_pertaining_to_illness]')
# 	sys.exit()

# if IS_RAND_FOREST:
# 	print('Using Random Forest Classifier')
# else:
# 	print('Using Logistic Regression Classifier')

# if INCLUDE_0:
# 	txt = 'including_0'
# else :
# 	txt = 'excluding_0'

# if IS_RAND_FOREST:
# 	model_name = 'random_forest'
# else:
# 	model_name = 'logistic_regression'

# # Random Forest Trained model takes too much disk space
# # Save model only if Logistic Regression is used
# # not if random forest is used
# to_save = True
# if IS_RAND_FOREST :
# 	to_save = False

# # Default file names
# dataset_save_path = str(int(state)) + '/' + str(state) + '_AHS_' + dtype
# clean_data_path = dataset_save_path + '_' + 'Clean.csv'

# if not INCLUDE_0 :
# 	diag_col_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_col' + '.csv'
# 	diag_hot_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_hotData' + '.csv'
# 	diag_cold_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_coldData' + '.csv'
# else :
	# diag_col_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_col' + '_with_0' + '.csv'
	# diag_hot_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_hotData' + '_with_0' + '.csv'
	# diag_cold_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_coldData' + '_with_0' + '.csv'


# diag_col_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_col' + '_with_0' + '.csv'
# diag_hot_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_hotData' + '_with_0' + '.csv'
# 	diag_cold_csv_name = dataset_save_path + '_' + pred_col[0:4] + '_coldData' + '_with_0' + '.csv'

# print('Dataset Used : ', diag_hot_csv_name)
# print('Labels Used : ', diag_col_csv_name)

# FINAL_RES = {}


data = pd.read_csv(data_path + "22/22_AHS_COMB_diag_hotData_with_0.csv")

# data = pd.read_csv(data_path + "22/22_AHS_COMB_diag_coldData_with_0.csv")
hot_data = data.drop(['state'], axis=1, errors='ignore')
hot_columns = list(hot_data.columns.values)
to_pred_col = pd.read_csv(data_path + "22/22_AHS_COMB_diag_col_with_0.csv")


# hot_data = pd.read_csv(data_path + diag_hot_csv_name)
# to_pred_col = pd.read_csv(data_path + diag_col_csv_name)
# print(hot_data.shape)
# # Dropping column 'state'
# hot_data = hot_data.drop(['state'], axis=1, errors='ignore')
# print(hot_data.shape)

# hot_columns = list(hot_data.columns.values)

# clf = IsolationForest()
# clf.fit(hot_data, to_pred_col)

clf = RandomForestClassifier()
clf = clf.fit(hot_data, to_pred_col)
model = SelectFromModel(clf, prefit=True)
hot_data1 = model.transform(hot_data)
new_cols = []
print(hot_data1.shape)
for i in range(hot_data1.shape[1]):
	# print i
	for col in hot_columns:
		# print col
		a = np.sum(np.abs(hot_data1[:,i]-hot_data[col]))
		
		if a == 0.0:
			print col
			new_cols.append(col)
			break
		
		# break
new_hot_data = pd.DataFrame(hot_data1, columns = new_cols)

# print new_hot_data.head()
print new_cols
# pd.DataFrame.to_csv(data_path+"22/22_AHS_COMB_diag_hotData_with_0_reduced.csv")
# hot_data_n = SelectKBest(chi2, 100).fit_transform(hot_data, to_pred_col)

# print(hot_columns)
# data = data.as_matrix()
# data_Y = np.reshape(data_Y.as_matrix(),(-1,))
# print data_Y.shape
# classif = mutual_info_classif(hot_data, to_pred_col)
# a = classif
# a = [0.00000000e+00,1.57466539e-03,1.42671832e-03,1.78906598e-03,1.35059652e-03,5.05637129e-04,2.37409189e-03,2.64643531e-03,6.67460117e-06,0.00000000e+00,0.00000000e+00,0.00000000e+00,3.76606246e-04,1.67398221e-03,0.00000000e+00,3.27777268e-03,3.41767869e-03,1.51477215e-03,0.00000000e+00,6.47972232e-03,0.00000000e+00,0.00000000e+00,3.13423337e-03,4.10685373e-03,0.00000000e+00,0.00000000e+00,0.00000000e+00,3.28639128e-03,1.30948672e-03,0.00000000e+00,1.73367244e-03,7.13473361e-04,7.15499334e-04,2.60301719e-03,0.00000000e+00,1.36968562e-03,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,2.63414809e-03,0.00000000e+00,0.00000000e+00,1.17043206e-04,0.00000000e+00,0.00000000e+00,7.58278369e-04,0.00000000e+00,0.00000000e+00,1.18561079e-04,1.47611989e-03,0.00000000e+00,5.16315884e-06,0.00000000e+00,0.00000000e+00,7.14175513e-04,8.60818475e-05,0.00000000e+00,0.00000000e+00,9.68752715e-04,0.00000000e+00,3.48933817e-04,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,1.74073171e-03,0.00000000e+00,0.00000000e+00,0.00000000e+00,2.97300827e-03,1.93986821e-04,0.00000000e+00,2.94527263e-04,6.42986394e-04,2.71343558e-03,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,4.28097227e-03,6.26379780e-04,0.00000000e+00,0.00000000e+00,3.53356854e-04,0.00000000e+00,0.00000000e+00,0.00000000e+00,1.84904346e-03,0.00000000e+00,3.60431554e-03,1.96307525e-03,2.54117559e-04,1.83092726e-04,1.89821390e-04,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,1.93466541e-03,2.71141610e-03,2.43175170e-03,0.00000000e+00,3.76964196e-03,0.00000000e+00,4.68709394e-04,0.00000000e+00,3.97940519e-03,0.00000000e+00,0.00000000e+00,7.30090602e-05,0.00000000e+00,0.00000000e+00,3.99153840e-03,1.25092134e-03,2.25206894e-03,9.15545176e-04,0.00000000e+00,0.00000000e+00,4.03090799e-04,6.40159410e-04,8.20949530e-04,0.00000000e+00,1.34639638e-03,3.01974294e-04,0.00000000e+00,1.43144527e-03,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,9.57696349e-04,0.00000000e+00,0.00000000e+00,0.00000000e+00,2.43060017e-03,0.00000000e+00,1.01612486e-03,1.48147629e-03,3.27611237e-03,2.65723279e-03,1.73170745e-03,7.36634716e-04,0.00000000e+00,7.06218533e-05,3.44727687e-04,0.00000000e+00,3.69931527e-03,0.00000000e+00,0.00000000e+00,0.00000000e+00,2.95988592e-04,5.98982493e-03,1.52773796e-03,4.03924387e-04,0.00000000e+00,6.10312280e-04,1.53066250e-03,3.39406596e-03,0.00000000e+00,1.63128616e-03,0.00000000e+00,2.11573438e-03,0.00000000e+00,7.63675529e-04,0.00000000e+00,2.05891162e-03,9.19566292e-04,6.52308369e-04,0.00000000e+00,2.27645346e-03,3.30244957e-04,3.89317663e-03,8.72075965e-04,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,2.30871651e-03,0.00000000e+00,3.74951101e-03,2.76704703e-03,0.00000000e+00,0.00000000e+00,4.33963788e-04,1.06310838e-03,2.39593937e-03,0.00000000e+00,1.21691060e-03,0.00000000e+00,0.00000000e+00,1.16984917e-03,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,1.48151179e-03,6.46361616e-04,2.70326666e-04,1.94691733e-03,1.51298155e-04,4.50037944e-04,0.00000000e+00,0.00000000e+00,3.94499216e-03,0.00000000e+00,8.44510471e-04,0.00000000e+00,2.95678372e-03,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,2.96931825e-03,3.61253755e-03,1.31782990e-03,2.59326537e-03,5.57594201e-04,6.46142704e-05,5.94347986e-03,5.04405037e-05,0.00000000e+00,0.00000000e+00,2.11640408e-03,1.19908313e-03,2.67779975e-03,0.00000000e+00,1.72423745e-03,7.13152955e-04,3.04602561e-03,6.63497712e-03,2.66102951e-03,2.83198805e-04,3.36000156e-03,0.00000000e+00,7.02415558e-03,0.00000000e+00,6.41868034e-03,2.20869534e-03,3.88207248e-03,0.00000000e+00,2.13558175e-03,0.00000000e+00,1.01434811e-03,1.37372553e-03,0.00000000e+00,3.28288712e-03,4.27718837e-03,1.10634315e-03,6.25810278e-04,1.43357908e-03,0.00000000e+00,7.12209742e-04,0.00000000e+00,0.00000000e+00,6.37568226e-04,0.00000000e+00,4.07631124e-04,2.41270365e-03,2.19471985e-03,0.00000000e+00,0.00000000e+00,1.59637698e-03,8.75053696e-04,0.00000000e+00,1.87982363e-04,0.00000000e+00,1.93540616e-03,0.00000000e+00,0.00000000e+00,2.85768578e-03,1.42664460e-03,1.53728398e-03,2.11208469e-03,4.11436272e-03,0.00000000e+00,1.86814183e-03,2.53604753e-04,2.16289986e-04,1.87278609e-04,3.47369082e-03,0.00000000e+00,0.00000000e+00,0.00000000e+00]
# f = [hot_columns[i] for i in range(len(a)) if a[i]==0.0]

# hot_data.drop(f, axis = 1, inplace = True)
# print (f)
# train_data, train_label, test_data, test_label, valid_data, valid_label = \
# 		split_data_in_train_test_valid(hot_data, to_pred_col, to_catg=False)

# model_path = model_save_path + str(pred_col) + '/' + txt + '/' + str(state) + '/' + \
# 										'full_data_mutual_reln' + '/' + 'sklearn' + '/'

# if not os.path.exists(model_path):	
# 	os.makedirs(model_path)

# log_reg = SKLearn_models(train_data, train_label, test_data, test_label, valid_data, 
# 	valid_label, model_path, model_name, save_model=to_save, rand_forest=IS_RAND_FOREST)

# log_reg.classify()






