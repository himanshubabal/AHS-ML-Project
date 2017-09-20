import pandas as pd
import numpy as np
import math
import sys
import os
import argparse
import pickle
import operator

import tensorflow as tf
from keras.utils.np_utils import to_categorical

# data_path = '/home/physics/btech/ph1140797/AHS-ML-Project/data/'
data_path = '/home/physics/btech/ph1140797/scratch/AHS_data/'
NaN_TOLRENCE = 50.0			# In %

parser = argparse.ArgumentParser(description='Pass STATE in -state and DTYPE in -dtype')
parser.add_argument('-state', default = 22,  type=int, help = 'State for which data is to be processed -- 5, 8, 9, 10, 18, 20, 21, 22, 23')
parser.add_argument('-dtype', default = 'COMB',  type=str, help = 'Type of Dataset -- COMB, MORT, WOMEN, WPS, CAB')
parser.add_argument('-col', default = 'diagnosed_for',  type=str, help = 'Column to be predicted -- diagnosed_for, illness_type, symptoms_pertaining_to_illness')
args = parser.parse_args()

states_list = [5, 8, 9, 10, 18, 20, 21, 22, 23]
dtypes_list = ['COMB', 'MORT', 'WOMEN', 'WPS', 'CAB']
pred_col_list = ['diagnosed_for', 'illness_type', 'symptoms_pertaining_to_illness']

state = int(args.state)
dtype = str(args.dtype)
pred_col = str(args.col)

if state not in states_list :
	print('Please enter -state from [5, 8, 9, 10, 18, 20, 21, 22, 23]')
	sys.exit()
if dtype not in dtypes_list :
	print('Please enter -dtype from [COMB, MORT, WOMEN, WPS, CAB]')
	sys.exit()
if pred_col not in pred_col_list :
	print('Please enter -dtype from [diagnosed_for, illness_type, symptoms_pertaining_to_illness]')
	sys.exit()

dataset_name = str(state) + '_AHS_' + dtype
default_data_csv = dataset_name + '.csv'
clean_csv_name = dataset_name + '_Clean' + '.csv'
diag_col_csv_name = dataset_name + '_' + pred_col[0:4] + '_col' + '.csv'
diag_hot_csv_name = dataset_name + '_' + pred_col[0:4] + '_hotData' + '.csv'

print('Dataset Used : ', dataset_name)

from python_helper.sort_clean_data import lowercase_32Char_list
from python_helper.sort_clean_data import get_sheet_field_names
from python_helper.sort_clean_data import remove_yellow_fields
from python_helper.sort_clean_data import sort_dataset_state_dist_house
from python_helper.sort_clean_data import create_balanced_classes

# Make dataframe One-Hot Encoded
# one_hot_colnames -> Columns which we wanted to be
# 					  one hot encoded.
#				If none provide, it will encode all 
#				columns except district, age, state
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

# Check of dataframe has column 'Unnamed: 0'
# If present, remove it
def check_unnamed(dataframe):
	if 'Unnamed: 0' in list(dataframe):
		dataframe = dataframe.drop('Unnamed: 0',axis=1,errors='ignore')
	return dataframe

# These are the columns which I think are irrelevant in the analysis
# Feel free to add or remove entries 
col_to_be_removed = [
	'v104'
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

# It will return a dataframe which will conatin
# Columns of 'df' along with % of entries with
# NaN in them, like column 'age' has 40% Nan
#
# It will return List of Tuples in desending order
# [(Col_name, %_NaN_values), (), (), () ....]
def missing_values_table(df): 
	mis_val = 100 * df.isnull().sum()/len(df)
	mis_val_index = mis_val.index.tolist()
	mis_val_val = mis_val.tolist()
	val_dict = {}
	for i in range(len(mis_val_index)):
		val_dict[mis_val_index[i]] = mis_val_val[i]
	# Sort it by values
	val_dict = sorted(val_dict.items(), key=operator.itemgetter(1), reverse=True)
	return val_dict 

# It will return dataframe after removing 'yellow' columns
# It will also convert all entries into 'float'
# The entries which can't be converted to float, will be
# converted to NaN
def get_clean_data()
	file_path = data_path + clean_csv_name
	if not os.path.exists(file_path):
		field_list_pickle = data_path + dtype + '_fields_list' + '.pickle'
		field_list = list()

		# Check if lists of yellwo fields saved on disk
		if os.path.exists(field_list_pickle):
			with open(field_list_pickle, 'rb') as f:
				field_list = pickle.load(f)
		
		# If not, extrace yellow columns from .xlxs file
		else:
			AHS_struct_workbook = pd.ExcelFile(data_path + "Data_structure_AHS.xlsx")
			AHS_struct_sheets_names = AHS_struct_workbook.sheet_names
			field_list = lowercase_32Char_list(get_sheet_field_names(AHS_struct_workbook, dtype))
			del AHS_struct_workbook, AHS_struct_sheets_names
			with open(field_list_pickle, 'wb') as f:
				pickle.dump(field_list, f)

		assert len(field_list) != 0

		AHS_dtype = pd.read_csv(data_path + default_data_csv, sep="|")
		print('Shape : ', AHS_dtype.shape)

		print('Removing Yellow Fields from COMB Data')
		data_clean = remove_yellow_fields(AHS_dtype, field_list[0])
		print('Clean Shape : ', data_clean.shape)
		del AHS_dtype, field_list

		if 'v104' in list(data_clean):
			data_clean.drop(['v104'], inplace=True, axis=1, errors='ignore')
		assert ('v104' not in list(data_clean))

		# Convert data to float
		# Those entries which can't be converted, replace by NaN
		print('Making Data float type')
		data_clean = data_clean.apply(pd.to_numeric, errors='coerce', downcast='float')

		# Saving Clean data to disk
		print('Saving Clean Data to ' + clean_csv_name)
		data_clean.to_csv(data_path + clean_csv_name, index=False)

	else:
		data_clean = pd.read_csv(data_path + clean_csv_name)

	return(data_clean)

# Drop columns of dataframe which have NaN values
# more then NaN_TOLRENCE % in them
def drop_nan_col(dataframe):
	# Removing Columns with more then 50% NaN Values
	# nan_dict = missing_values_table(dist_p)
	nan_dict = missing_values_table(dataframe)
	nan_col = list()

	for col in nan_dict:
		if col[1] > NaN_TOLRENCE :
			nan_col.append(col[0])
	print('Nan Col : ', nan_col)

	# dist_p = dist_p.drop(nan_col, axis=1, errors='ignore')
	dataframe = dataframe.drop(nan_col, axis=1, errors='ignore')
	print('After Dropping Nan Col Shape : ', dataframe.shape)
	return(dataframe)

# Set drop_col_with_almost_nan = True
# if you want columns with more then NaN_TOLRENCE %
# of Nan values to be removed from dataframe
# NaN are by default replaced by 0, and it doesn't have
# much difference on accuracy, so False by default
#
# PRED_COLNAME -> Name to column which we want to predict
# Fetched from argument parser, defalut = 'diagnosed_for'
def prep_for_analysis(drop_col_with_almost_nan=False, PRED_COLNAME=pred_col):
	data_clean = get_clean_data()
	print('Clean Data shape : ', data_clean.shape)
	print('  ')

	# Checking if to be predicted col is in the dataframe
	assert(PRED_COLNAME in list(data_clean))

	# Drop the above mentioned columns
	print('Removing not-so-useful columns')
	data_clean = data_clean.drop(col_to_be_removed, axis=1, errors='ignore')
	print('Shape after removing : ', data_clean.shape)
	
	# Drop rows with 'NaN' for to-be-predicted column
	data_clean = data_clean.dropna(subset=[PRED_COLNAME])
	print('Drop NAN shape : ', data_clean.shape)

	# Balance the dataframe labels, So as 
	# every label contains almost equal data
	data_clean = create_balanced_classes(data_clean, [1.0,2.0,3.0,7.0,9.0,19.0,21.0,99.0],PRED_COLNAME)
	print('After balance : ', data_clean.shape)

	# If drop_col_with_almost_nan = True
	# Drop Col with Nan more then tolerence
	if drop_col_with_almost_nan:
		data_clean = drop_nan_col(data_clean)

	data_clean = data_clean.apply(pd.to_numeric, errors='coerce', downcast='float')
	data_clean = data_clean.fillna(0)
	data_clean = data_clean.astype(float)

	data_clean = data_clean.iloc[np.random.permutation(len(data_clean))]
	data_clean = data_clean.reset_index(drop=True)

	print('Splitting to-predict column')
	# Seperating 'diagnosed_for' variable for prediction
	diagnosed_col = data_clean[[PRED_COLNAME]]
	diagnosed_data = data_clean.drop([PRED_COLNAME], inplace=False, axis=1, errors='ignore')
	del data_clean

	print('  ')
	print('Saving to-be-predicted column to ' + diag_col_csv_name)
	diagnosed_col = check_unnamed(diagnosed_col)
	diagnosed_col.to_csv(data_path + diag_col_csv_name, index=False)
	del diagnosed_col

	print('One Hot Encoding Data')
	try:
		diagnosed_data = one_hot_df(diagnosed_data)
	except Exception as e:
		# One - Hot encoding for the data
		# One Hot encoding of very large datasets is very memory consuming,
		# So, we will split large datasets into small ones
		# Split dataframe into smaller ones containing 30,000 rows each
		print('One Hot Encoding by splitting into smaller chunks')
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

	print('Hot Encoded shape : ', diagnosed_data.shape)
	print('Saving One-Hot Columns to ' + diag_hot_csv_name)
	diagnosed_data = check_unnamed(diagnosed_data)
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

check_if_preped()