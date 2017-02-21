import tensorflow as tf

# import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt

import math
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

data_path = 'data/'


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
	# mort_clean.to_csv(data_path + '22_AHS_MORT_Clean.csv')
	# wps_clean.to_csv(data_path + '22_AHS_WPS_Clean.csv')
	# women_clean.to_csv(data_path + '22_AHS_WOMEN_Clean.csv')

	comb_field_list = lowercase_32Char_list(get_sheet_field_names(AHS_struct_workbook, "COMB"))
	AHS_comb = pd.read_csv(data_path + "22_AHS_COMB.csv", sep="|")

	print('    ')
	print('Removing Yellow Fields from COMB Data')
	data_clean = remove_yellow_fields(AHS_comb, comb_field_list[0])

	print('Saving Clean Data to data/22_AHS_COMB_Clean.csv')
	data_clean.to_csv(data_path + '22_AHS_COMB_Clean.csv')

	del AHS_struct_workbook, AHS_struct_sheets_names
	del AHS_comb, data_clean, comb_field_list

def check_if_exists(force=False):
	file_path = data_path + '22_AHS_COMB_Clean.csv'
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
	dist = pd.read_csv(data_path + '22_AHS_COMB_Clean.csv')

	print('  ')
	print('Removing not-so-useful columns')
	# Dropping the above columns
	dist = dist.drop(col_to_be_removed,axis=1,errors='ignore')

	# As we need to calculate for variable 'diagnosed_for'
	# So, we drop the rows where 'diagnosed_for' == NaN
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
	print('Saving diagnosed_for column to data/22_COMB_diag_col.csv')
	diagnosed_col.to_csv(data_path + '22_COMB_diag_col.csv')
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


	print('Saving One-Hot Columns to data/22_COMB_diag_hotData.csv')
	diagnosed_data.to_csv(data_path + '22_COMB_diag_hotData.csv')
	del diagnosed_data

def check_if_exists(force=False):
	file_path_1 = data_path + '22_COMB_diag_col.csv'
	file_path_2 = data_path + '22_COMB_diag_hotData.csv'
	if (not os.path.exists(file_path_1)) and (not os.path.exists(file_path_2)):
		prep_for_analysis()
	else:
		if force:
			prep_for_analysis()
		else:
			print('Data already Processed. Moving to the next part')

# Set force = True  to force it to redo even if it exists
check_if_exists()

print('------------------------------------------------------')
print('    ')

################### PART - 3 : Apply Machine Learning on the data ###################
#####################################################################################

print('PART - 3 : Applying Machine Learning on the data')

diagnosed_data = pd.read_csv(data_path + '22_COMB_diag_hotData.csv', low_memory=False)
diagnosed_col = pd.read_csv(data_path + '22_COMB_diag_col.csv', low_memory=False)

if 'Unnamed: 0' in list(diagnosed_col):
	diagnosed_col = diagnosed_col.drop('Unnamed: 0',axis=1,errors='ignore')

assert (diagnosed_data.shape[0] == diagnosed_col.shape[0])
split_index = int(diagnosed_data.shape[0] * 0.85)

print('   ')
print('Splitting train and test data in ratio 85:15')
train_data = np.array(diagnosed_data.astype(float))[:split_index]
train_label = np.array(diagnosed_col.astype(float))[:split_index][:,0]

test_data = np.array(diagnosed_data.astype(float))[split_index:]
test_label = np.array(diagnosed_col.astype(float))[split_index:][:,0]


# Replace Label No 99 by 32
# Label No 99 causes 'to_categorical' to make 100 one-hot values
# Replacing it by 33 leads to only 33 values
def replace_99_labes(label_data):
	for i in range(len(label_data)):
		if label_data[i] == 99.0 :
			label_data[i] = 32.0

	return label_data

train_rep = replace_99_labes(train_label)
test_rep = replace_99_labes(test_label)

train_label = to_categorical(train_rep.astype('int32'), nb_classes=None)
test_label = to_categorical(test_rep.astype('int32'), nb_classes=None)


# Fully-Connected Neural network with 4 Hidden layers
model = Sequential()
# Input Layer
model.add(Dense(1000, input_dim=test_data.shape[1], init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Hidden Layer - 1
model.add(Dense(750, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Hidden Layer - 2
model.add(Dense(500, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Hidden Layer - 3
model.add(Dense(250, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Hidden Layer - 4
model.add(Dense(100, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Output Layer
model.add(Dense(test_label.shape[1], init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


model.fit(train_data, train_label,
          nb_epoch=200,
          batch_size=128)

model.evaluate(test_data, test_label)
