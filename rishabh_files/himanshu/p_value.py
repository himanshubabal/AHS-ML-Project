from sklearn.feature_selection import chi2
import pandas as pd
from keras.utils.np_utils import to_categorical
import numpy as np
# from ..variables import data_path, model_save_path, data_scratch
from models.sklearn_classifiers import SKLearn_models

data_path = '/home/physics/btech/ph1140797/AHS-ML-Project/data/'
model_save_path = '/home/physics/btech/ph1140797/AHS-ML-Project/saved_models/'

def find_hot_matches(col_index_list_to_remove, hot_col_names_list):
    hot_index_list = list()

    for cold in col_index_list_to_remove:
        for hot in hot_col_names_list:
            if cold in hot:
                index = hot_col_names_list.index(hot)
                if index not in hot_index_list :
                    hot_index_list.append(index)
    return (hot_index_list)

def replace_labels(label_data):
		# if 0.0 not in np.unique(label_data):
		# 	process_0 = False
		dict_map = {0.0 : 0.0, 1.0 : 1.0, 2.0 : 2.0, 3.0 : 3.0, 7.0 : 4.0, 9.0 : 5.0,
					19.0 : 6.0, 21.0 : 7.0, 99.0 : 7.0}
		cols = list(label_data)
		data = np.array(label_data.astype(float))[:,0]
		for i in range(len(label_data)):
			# element = label_data.iat[i,0]
			if data[i] in dict_map:
				data[i] = dict_map[data[i]]
				# if process_0:
				# 	data[i] = dict_map[float(label_data.iat[i,0])]
				# else:
				# 	data[i] = dict_map[float(label_data.iat[i,0])]-1
		data = data[:,np.newaxis]
		return data

def split_data_in_train_test_valid(diagnosed_data, diagnosed_col, to_catg=True):
	assert (diagnosed_data.shape[0] == diagnosed_col.shape[0])

	# Split train-test-valid in ratio 60:25:15
	split_train = int(diagnosed_data.shape[0] * 0.60)
	split_test = split_train + int(diagnosed_data.shape[0] * 0.25)

	# Convert dataset from pd dataframe to numpy arrays for further processing
	diagnosed_data = np.array(diagnosed_data.astype(float))

	# Replace certain labels and then one-hot encode 'diagnosed_for' column
	diagnosed_col = replace_labels(diagnosed_col)
	diagnosed_col = np.array(diagnosed_col.astype(float))[:,0]

	if to_catg:
		diagnosed_col = to_categorical(diagnosed_col.astype('int32'))

	# Split train, test and validation datasets
	train_data = diagnosed_data[:split_train]
	train_label = diagnosed_col[:split_train]

	test_data = diagnosed_data[split_train:split_test]
	test_label = diagnosed_col[split_train:split_test]

	valid_data = diagnosed_data[split_test:]
	valid_label = diagnosed_col[split_test:]

	print('Train dataset : ', train_data.shape, train_label.shape)
	print('Test dataset : ', test_data.shape, test_label.shape)
	print('Validation Dataset : ', valid_data.shape, valid_label.shape)

	return(train_data, train_label, test_data, test_label, valid_data, valid_label)

columns = open(data_path + "22/22_AHS_COMB_diagnosed_for_col_names.csv","r").readlines()[1:]
columns = [col.strip() for col in columns]

data = pd.read_csv(data_path + "22/22_AHS_COMB_diag_hotData_with_0.csv")

# data = pd.read_csv(data_path + "22/22_AHS_COMB_diag_coldData_with_0.csv")
data = data.drop(['state'], axis=1, errors='ignore')
hot_columns = list(data.columns.values)
data_Y = pd.read_csv(data_path + "22/22_AHS_COMB_diag_col_with_0.csv")

data = data.as_matrix()
data_Y = data_Y.as_matrix()

ch_X = chi2(data, data_Y)


train_data, train_label, test_data, test_label, valid_data, valid_label = \
		split_data_in_train_test_valid(data, data_Y, to_catg=False)

model_path = model_save_path + "diagnosed_for" + '/' + "including_0" + '/' + "22" + '/' + \
												'full_data' + '/' + 'sklearn' + '/'

log_reg = SKLearn_models(train_data, train_label, test_data, test_label, valid_data, 
	valid_label, model_path, "logistic_regression", save_model=False, rand_forest=True)

model = log_reg.get_model()


# print ch_X
p_values = model.feature_importances_
print len(p_values)
# for j in range(p_values.shape[0]):
	# print "j th  class\n\n\n"
for col in columns:
	# print col, len(hot_columns), hot_columns
	indices = find_hot_matches([col], hot_columns)
	# print indices
	if len(indices) > 0:
		p_val_arr = [p_values[i] for i in indices]
		print p_val_arr
		print col
		print sum(p_val_arr)/len(p_val_arr)
		print "----------------------------------------\n"
		# break

# print ch_X
print len(p_values)
