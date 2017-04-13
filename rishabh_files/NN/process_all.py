import tensorflow as tf

import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import math
import os
import argparse
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

data_path = '/home/physics/btech/ph1140797/AHS-ML-Project/data/'

parser = argparse.ArgumentParser(description='Pass STATE in -state and DTYPE in -dtype')
parser.add_argument('-state', default = 22,  type=int, help = 'State for which data is to be processed -- 5, 8, 9, 10, 18, 20, 21, 22, 23')
parser.add_argument('-dtype', default = 'COMB',  type=str, help = 'Type of Dataset -- COMB, MORT, WOMEN, WPS, CAB')
args = parser.parse_args()

states_list = [5, 8, 9, 10, 18, 20, 21, 22, 23]
dtypes_list = ['COMB', 'MORT', 'WOMEN', 'WPS', 'CAB']

# state = int(parser.parse_args(['-state']))
# dtype = str(parser.parse_args(['-dtype']))
state = int(args.state)
dtype = str(args.dtype)

if state not in states_list :
	print('Please enter -state from [5, 8, 9, 10, 18, 20, 21, 22, 23]')
	sys.exit()
if dtype not in dtypes_list :
	print('Please enter -dtype from [COMB, MORT, WOMEN, WPS, CAB]')
	sys.exit()

dataset_name = str(state) + '_AHS_' + dtype
default_data_csv = dataset_name + '.csv'
clean_csv_name = dataset_name + '_Clean' + '.csv'
diag_col_csv_name = dataset_name + '_diag_col' + '.csv'
diag_hot_csv_name = dataset_name + '_diag_hotData' + '.csv'

print('Dataset Used : ', dataset_name)

print('------------------------------------------------------')
print('PART - 1 : Remove Yellow Field Columns form CSV File')

############# PART - 1 : Remove Yellow Field Columns form CSV File #############
################################################################################

from python_helper.sort_clean_data import lowercase_32Char_list
from python_helper.sort_clean_data import get_sheet_field_names
from python_helper.sort_clean_data import remove_yellow_fields
from python_helper.sort_clean_data import sort_dataset_state_dist_house
from python_helper.sort_clean_data import create_balanced_classes

def remove_yellow_df():
	AHS_struct_workbook = pd.ExcelFile(data_path + "Data_structure_AHS.xlsx")
	AHS_struct_sheets_names = AHS_struct_workbook.sheet_names

	# ---- Uncomment these lines for processing other datasets as well ----
	# mort_field_list = lowercase_32Char_list(get_sheet_field_names(AHS_struct_workbook, "MORT"))
	# wps_field_list = lowercase_32Char_list(get_sheet_field_names(AHS_struct_workbook, "WPS"))
	# women_field_list = lowercase_32Char_list(get_sheet_field_names(AHS_struct_workbook, "WOMAN"))
	#
	# AHS_mort = pd.read_csv(data_path + "22_AHS_MORT.csv", sep="|")
	# AHS_wps = pd.read_csv(data_path + "22_AHS_WPS.csv", sep="|")
	# AHS_women = pd.read_csv(data_path + "22_AHS_WOMEN.csv", sep="|")
	#
	# mort_clean = remove_yellow_fields(AHS_mort, mort_field_list[0])
	# wps_clean = remove_yellow_fields(AHS_wps, wps_field_list[0])
	# women_clean = remove_yellow_fields(AHS_women, women_field_list[0])
	#
	# mort_clean.to_csv(data_path + '22_AHS_MORT_Clean.csv', index=False)
	# wps_clean.to_csv(data_path + '22_AHS_WPS_Clean.csv', index=False)
	# women_clean.to_csv(data_path + '22_AHS_WOMEN_Clean.csv', index=False)

	# comb_field_list = lowercase_32Char_list(get_sheet_field_names(AHS_struct_workbook, "COMB"))
	# AHS_comb = pd.read_csv(data_path + "22_AHS_COMB.csv", sep="|")
	# field_list = lowercase_32Char_list(get_sheet_field_names(AHS_struct_workbook, dtype))

	field_list_pickle = data_path + dtype + '_fields_list' + '.pickle'
	field_list = list()

	if os.path.exists(field_list_pickle):
		with open(field_list_pickle, 'rb') as f:
			field_list = pickle.load(f)
		
	else:
		field_list = lowercase_32Char_list(get_sheet_field_names(AHS_struct_workbook, dtype))
		
    	with open(field_list_pickle, 'wb') as f:
    		pickle.dump(field_list, f)


    assert (len(field_list) != 0)

	AHS_dtype = pd.read_csv(data_path + default_data_csv, sep="|")

	print('    ')
	print('Removing Yellow Fields from COMB Data')
	data_clean = remove_yellow_fields(AHS_dtype, field_list[0])

	print('Saving Clean Data to data/' + clean_csv_name)
	# data_clean.to_csv(data_path + '22_AHS_COMB_Clean.csv', index=False)
	data_clean.to_csv(data_path + clean_csv_name, index=False)

	del AHS_struct_workbook, AHS_struct_sheets_names
	del AHS_dtype, data_clean, field_list

def check_if_exists(force=False):
	file_path = data_path + clean_csv_name
	if not os.path.exists(file_path):
		remove_yellow_df()
	else:
		if force:
			remove_yellow_df()
		else:
			print('Yellow Fields already removed. Proceeding further')

# Set force = True  to force it to redo even if it exists
check_if_exists()

print('------------------------------------------------------')

################### PART - 2 : Prepare COMB Data for the analysis ###################
#####################################################################################

print('PART - 2 : Preparing Data for the analysis')

def one_hot_df(data_frame, one_hot_colnames=list()) :
    if len(one_hot_colnames) != 0:
        colnames = list(data_frame)
        hot_col = list()
        for hot in one_hot_colnames :
            if hot in colnames :
                hot_col.append(hot)
    else:
        hot_col = list(data_frame)
    if 'district' in hot_col :
        hot_col.remove('district')
    if 'state' in hot_col :
        hot_col.remove('state')
    if 'age' in hot_col:
        hot_col.remove('age')     
    data_frame = pd.get_dummies(data_frame, columns=hot_col)
    return (data_frame)


# These are the columns which I think are irrelevant in the analysis
# Feel free to add or remove entries 
col_to_be_removed = [
    'state',
    'Unnamed: 0',
    'psu_id',
    'house_no',
    'house_hold_no',
    'member_identity',
    'father_serial_no',
    'mother_serial_no',
    'date_of_birth',
    'month_of_birth',
    'year_of_birth',
    'date_of_marriage',
    'month_of_marriage',
    'year_of_marriage',
    'building_no',
    'no_of_dwelling_rooms',
    'rural_1',
    'rural_2',
    'stratum_code',
    'relation_to_head',
    'member_identity',
    'father_serial_no',
    'mother_serial_no',
    'date_of_birth',
    'month_of_birth',
    'year_of_birth',
    'date_of_marriage',
    'month_of_marriage',
    'year_of_marriage',
    'isheadchanged',
    'year'
]


def prep_for_analysis():
	dist = pd.read_csv(data_path + clean_csv_name)

	print('  ')
	print('Removing not-so-useful columns')
	# Dropping the above columns
	dist = dist.drop(col_to_be_removed, axis=1, errors='ignore')

	# As we need to calculate for variable 'diagnosed_for'
	# So, we drop the rows where 'diagnosed_for' == NaN

	#-#-#-#-#-# We can also use data where rows 'diagnosed_for' == NaN
	#-#-#-#-#-# to predict  and use as real world data

	dist_p = dist[np.isfinite(dist['diagnosed_for'])]
	del dist
	# dist_p = dist_p.reset_index(drop=True)

	# Removing rows with 'diagnosed_for' = 0.0
	# dist_p = dist_p[dist_p['diagnosed_for'] != 0.0]
	dist_p = create_balanced_classes(dist_p, [1.0,2.0,3.0,7.0,9.0,19.0,20.0,99.0],"diagnosed_for")
	# dist_p = dist_p.reset_index(drop=True)

	# Shuffling the dataset and reset index
	dist_p = dist_p.iloc[np.random.permutation(len(dist_p))]
	dist_p = dist_p.reset_index(drop=True)

	print('Splitting to-predict column')
	# Seperating 'diagnosed_for' variable for prediction
	diagnosed_col = dist_p[['diagnosed_for']]
	diagnosed_data = dist_p.drop(['diagnosed_for'], inplace=False, axis=1, errors='ignore')
	del dist_p

	print('  ')
	print('Saving diagnosed_for column to data/' + diag_col_csv_name)
	diagnosed_col.to_csv(data_path + diag_col_csv_name, index=False)
	del diagnosed_col

	print('One Hot Encoding Data')
	# One - Hot encoding for the data
	# One Hot encoding of very large datasets is very memory consuming,
	# So, we will split large datasets into small ones
	# Split dataframe into smaller ones containing 30,000 rows each
	size_threshold = 30000
	no_of_df = int(diagnosed_data.shape[0]/size_threshold)
	# List of splitted datasets
	splitted_dataset = np.array_split(diagnosed_data, no_of_df)
	df_list = list()
	for df in splitted_dataset:
		df_list.append(pd.DataFrame(df))

	hot_df_list = list()
	for df in df_list:
		hot_df_list.append(one_hot_df(df))

	diagnosed_data = pd.concat(hot_df_list)
	diagnosed_data = diagnosed_data.fillna(0)


	print('Saving One-Hot Columns to data/' + diag_hot_csv_name)
	diagnosed_data.to_csv(data_path + diag_hot_csv_name, index=False)
	del diagnosed_data

def check_if_preped(force=False):
	file_path_1 = data_path + diag_col_csv_name
	file_path_2 = data_path + diag_hot_csv_name
	if (not os.path.exists(file_path_1)) and (not os.path.exists(file_path_2)):
		prep_for_analysis()
	else:
		if force:
			prep_for_analysis()
		else:
			print('Data already Processed. Moving to the next part')

# Set force = True  to force it to redo even if it exists
check_if_preped()

def check_if_colnames_onDisk(force=False):
	file_path = data_path + 'diagnosed_for_col_names.csv'
	if ((not os.path.exists(file_path)) or force):
		dist = pd.read_csv(data_path + '22_AHS_COMB_Clean.csv', nrows=5)
		dist = dist.drop(col_to_be_removed, axis=1, errors='ignore')

		pd.DataFrame(list(dist)).to_csv(data_path + 'diagnosed_for_col_names.csv', index=False)
		del dist

check_if_colnames_onDisk()
print('------------------------------------------------------')

################### PART - 3 : Apply Machine Learning on the data ###################
#####################################################################################

print('PART - 3 : Dividing Data in train-test split and making categorical variables')

def replace_labes(label_data):
	    dict_map = {0.0 : 0.0, 1.0 : 1.0, 2.0 : 2.0, 3.0 : 3.0, 7.0 : 4.0, 9.0 : 5.0,
	                19.0 : 6.0, 21.0 : 7.0, 99.0 : 7.0}
	    for i in range(len(label_data)):
	        if label_data[i] in dict_map:
	            label_data[i] = dict_map[label_data[i]]-1
	        else :
	            label_data[i] = 0.0
	    return label_data

def check_unnamed(dataframe):
	if 'Unnamed: 0' in list(dataframe):
		dataframe = dataframe.drop('Unnamed: 0',axis=1,errors='ignore')
	return dataframe

def split_data():
	diagnosed_data = pd.read_csv(data_path + diag_hot_csv_name, low_memory=False)
	diagnosed_col = pd.read_csv(data_path + diag_col_csv_name, low_memory=False)

	diagnosed_col = check_unnamed(diagnosed_col)
	diagnosed_data = check_unnamed(diagnosed_data)
	# if 'Unnamed: 0' in list(diagnosed_col):
	# 	diagnosed_col = diagnosed_col.drop('Unnamed: 0',axis=1,errors='ignore')
	# if 'Unnamed: 0' in list(diagnosed_data):
	# 	diagnosed_data = diagnosed_data.drop('Unnamed: 0',axis=1,errors='ignore')

	assert (diagnosed_data.shape[0] == diagnosed_col.shape[0])
	split_index = int(diagnosed_data.shape[0] * 0.75)

	print('   ')
	print('Splitting train and test data in ratio 75:25')
	train_data = np.array(diagnosed_data.astype(float))[:split_index]
	train_label = np.array(diagnosed_col.astype(float))[:split_index][:,0]

	test_data = np.array(diagnosed_data.astype(float))[split_index:]
	test_label = np.array(diagnosed_col.astype(float))[split_index:][:,0]

	train_rep = replace_labes(train_label)
	test_rep = replace_labes(test_label)

	train_label = to_categorical(train_rep.astype('int32'), nb_classes=None)
	test_label = to_categorical(test_rep.astype('int32'), nb_classes=None)

	print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
	del diagnosed_data, diagnosed_col

	print('Saving test and train data')
	# train_data.to_csv(data_path + '22_COMB_diag_train_data.csv', index=False)
	# test_data.to_csv(data_path + '22_COMB_diag_test_data.csv', index=False)
	# train_label.to_csv(data_path + '22_COMB_diag_train_label.csv', index=False)
	# test_label.to_csv(data_path + '22_COMB_diag_test_label.csv', index=False)
	np.save(data_path + dataset_name + '_diag_train_data.npy', train_data)
	np.save(data_path + dataset_name + '_diag_test_data.npy', test_data)
	np.save(data_path + dataset_name + '_diag_train_label.npy', train_label)
	np.save(data_path + dataset_name + '_diag_test_label.npy', test_label)

	del train_data, train_label
	del test_data, test_label

def check_if_splitted(force=False):
	file_path_1 = data_path + dataset_name + '_diag_train_data.npy'
	file_path_2 = data_path + dataset_name + '_diag_test_data.npy'
	file_path_3 = data_path + dataset_name + '_diag_train_label.npy'
	file_path_4 = data_path + dataset_name + '_diag_test_label.npy'
	if ((not os.path.exists(file_path_1)) and (not os.path.exists(file_path_2)) 
		and (not os.path.exists(file_path_3)) and (not os.path.exists(file_path_4))):
		split_data()
	else:
		if force:
			split_data()
		else :
			print('Data already splitted')
		
		
check_if_splitted()

# train_data = np.load(data_path + dataset_name + 'diag_train_data.npy')
# test_data = np.load(data_path + dataset_name + 'diag_test_data.npy')
# train_label = np.load(data_path + dataset_name + 'diag_train_label.npy')
# test_label = np.load(data_path + dataset_name + 'diag_test_label.npy')








