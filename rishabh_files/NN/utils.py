import numpy as np
import pandas as pd
import cPickle as pickle

def read_data(data_path):
    diagnosed_data = pd.read_csv(data_path + '22_COMB_diag_hotData.csv', low_memory=False)
    diagnosed_col = pd.read_csv(data_path + '22_COMB_diag_col.csv', low_memory=False)
    out_col_names = [str(i) for i in list(diagnosed_col.diagnosed_for.unique())]

    if 'Unnamed: 0' in list(diagnosed_col):
        diagnosed_col = diagnosed_col.drop('Unnamed: 0',axis=1,errors='ignore')
    if 'Unnamed: 0' in list(diagnosed_data):
        diagnosed_data = diagnosed_data.drop('Unnamed: 0',axis=1,errors='ignore')
    columns = list(diagnosed_data.columns.values)
    assert (diagnosed_data.shape[0] == diagnosed_col.shape[0])
    split_index = int(diagnosed_data.shape[0] * 0.75)

    print('   ')
    print('Splitting train and test data in ratio 75:25')
    # train_data, test_data, train_label, test_label = train_test_split(np.array(diagnosed_data.astype(float)), diagnosed_col.astype(float), test_size = 0.3)
    train_data = np.array(diagnosed_data.astype(float))[:split_index]
    train_label = np.array(diagnosed_col.astype(float))[:split_index][:,0]
    # train_data = np.array(diagnosed_data.astype(float))
    # train_label = np.array(diagnosed_col.astype(float))[:,0]

    test_data = np.array(diagnosed_data.astype(float))[split_index:]
    test_label = np.array(diagnosed_col.astype(float))[split_index:][:,0]
    train_rep = replace_labes(train_label)
    test_rep = replace_labes(test_label)
    train_label = to_categorical(train_rep.astype('int32'), nb_classes=7)
    test_label = to_categorical(test_rep.astype('int32'), nb_classes=7)
    return train_data, test_data





def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, nb_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def replace_labes(label_data):
    dict_map = {0.0:0.0,
                1.0:1.0,
                2.0:2.0,
                3.0:3.0,
                7.0:4.0,
                9.0:5.0,
                19.0:6.0,
                21.0:7.0,
                99.0:7.0}
    for i in range(len(label_data)):
        if label_data[i] in dict_map:
            label_data[i] = dict_map[label_data[i]]-1
        else :
            label_data[i] = 0.0
    return label_data

def read_red_data(data_path,red_data_file, dims = 200):
    diagnosed_data = pickle.load(open(red_data_file, "r"))
    diagnosed_col = pd.read_csv(data_path + '22_COMB_diag_col.csv', low_memory=False)
    if 'Unnamed: 0' in list(diagnosed_col):
        diagnosed_col = diagnosed_col.drop('Unnamed: 0',axis=1,errors='ignore')
    split_index = int(diagnosed_data.shape[0] * 0.75)
    train_data = diagnosed_data[:split_index]
    test_data = diagnosed_data[split_index:]
    train_label = np.array(diagnosed_col.astype(float))[:split_index][:,0]
    test_label = np.array(diagnosed_col.astype(float))[split_index:][:,0]
    train_rep = replace_labes(train_label)
    test_rep = replace_labes(test_label)
    train_label = to_categorical(train_rep.astype('int32'), nb_classes=7)
    test_label = to_categorical(test_rep.astype('int32'), nb_classes=7)
    return train_data, train_label, test_data, test_label
