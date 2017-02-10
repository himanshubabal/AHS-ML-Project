import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import math

# Comb data is located in '3'
dataset_path = '/Users/himanshubabal/Documents/External_Disk_Link_WD_HDD/AHS_Data/22/3/'

# Make Data-frame One-Hot encoded for given column names list
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
            
    df = pd.get_dummies(data_frame, columns=hot_col)
    return (df)

dist = pd.read_csv('Data/3_22_8.csv')
# Dropping useless column 'Unnamed: 0'
dist = dist.drop(dist.columns[[0]], axis=1) 

# dist.shape
# (97824, 77)

# Columns that seem to have no significant effect on outcome
# These should be removed before doing the analysis
col_to_be_removed = [
    'state',
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

# Dropping the above columns
dist_n = dist.drop(col_to_be_removed,axis=1,errors='ignore')

# As we need to calculate for variable 'diagnosed_for'
# So, we drop the rows where 'diagnosed_for' == NaN
dist_p = dist_n[np.isfinite(dist_n['diagnosed_for'])]
# 82830 left out of 97824 (85%)
# Lost 15% data

# Reseting the index as it was distrubed by the removel of 'NaN' values
# Index reset causes and extra column named 'index' containing old indexes, so removing that also
# dist_p = pd.DataFrame.reset_index(dist_p).drop(['index'],inplace=False,axis=1,errors='ignore')
dist_p = dist_p.reset_index(drop=True)

# Dropping 'pus_id' from main dataframe and storing it in a seperate dataframe
dist_psu = dist_p[['psu_id']]
dist_p.drop(['psu_id'],inplace=True,axis=1,errors='ignore')

# dist_p.shape
# (82830, 57)

# dist_p[["illness_type", "diagnosed_for"]].groupby(["illness_type", "diagnosed_for"]).count()
# dist_p.groupby(["illness_type", "diagnosed_for"]).size()
# It shows that - 
#           Elements with both 'illness_type' = 0 and 'diagnosed_for' = 0
#           i.e. No illness and Not Diagnosed
#           There are '67250' such  cases out of 82830 (81%)
#               --> 81% of the data is useless for the analysis of 'diagnosed_for'

# It will give frequency of values for a specific feature
# Set Flag 'single_col_df' -> True if dataframe contains only single column
#    eg. For the values dataframe, leave colname -> ''
#     -> get_frequency(diagnosed_for_df, '', single_col_df=True)
def get_frequency(dataframe, colname, single_col_df=False) :
    # Get the specific column for which we need frequency
    if not single_col_df :
        dataframe = dataframe[[colname]]
    # Get the frequency in unsorted manner
    value_df = dataframe.apply(pd.value_counts)
    # Sort it
    sorted_values = pd.DataFrame.sort(value_df)
    return (sorted_values)

# get_frequency(dist_p, 'illness_type')
# Frequency of 'illness_type'
# illness_type
# 0.0       70230
# 1.0       1372
# 2.0       293
# 3.0       3208
# 4.0       118
# 5.0       2973
# 6.0       224
# 7.0       3693
# 8.0       22
# 9.0       697
######## Nealy 85% have illness_type -> 0.0, i.e. No illness

# get_frequency(dist_p, 'diagnosed_for')
# Frequency for 'diagnosed_for'
# diagnosed_for
# 0.0       79071
# 1.0       330
# 2.0       311
# 3.0       132
# 4.0       42
# 5.0       9
# 6.0       50
# 7.0       724
# 8.0       38
# 9.0       200
# 10.0      63
# 11.0      55
# 12.0      45
# 13.0      68
# 14.0      8
# 15.0      10
# 16.0      6
# 17.0      8
# 18.0      175
# 19.0      450
# 20.0      44
# 21.0      442
# 23.0      1
# 24.0      1
# 25.0      3
# 26.0      3
# 27.0      20
# 28.0      2
# 29.0      2
# 30.0      20
# 31.0      222
# 99.0      275
######## Nealy 95% have diagnosed_for -> 0.0, i.e. Not Diagnosed


# Removing rows with 'diagnosed_for' = 0.0
dist_diag = dist_p[dist_p['diagnosed_for'] != 0.0]
# dist_diag.shape -> (3759, 58)  -> 4.0%

# Removing rows with 'illness_type' = 0.0
dist_ill = dist_p[dist_p['illness_type'] != 0.0]
# dist_ill.shape -> (12600, 58)   -> 13.0%

# Removing rows with 'diagnosed_for' = 0.0 and 'illness_type' = 0.0
dist_diag_ill = dist_diag[dist_diag['illness_type'] != 0.0]
# dist_diag_ill.shape -> (779, 58)

## It seems like we have only ___ 0.8% ___ of the original data to work with
## We Lost ___ 99.2% ___ of the original data in processing
## So, we will need to do analysis on state as a whole. 
## District data in itself is very small, so neural net trained on it won't be such a good thing


# Since we need to predict 'diagnosed_for'
# So, seperating it from main dataframe and storing in a seperate dataframe
dist_diag_ill = dist_diag_ill.reset_index(drop=True)
diagnosed_col = dist_diag_ill[['diagnosed_for']]
diagnosed_data = dist_diag_ill.drop(['diagnosed_for'],inplace=False,axis=1,errors='ignore')
# One - Hot encoding for the data
diagnosed_data_hot = one_hot_df(diagnosed_data)

# Test and Train Split
train_data = np.array(diagnosed_data_hot.astype(float))[:700]
label_data = np.array(diagnosed_col.astype(float))[:700][:,0]

test_data = np.array(diagnosed_data_hot.astype(float))[700:]
test_label = np.array(diagnosed_col.astype(float))[700:][:,0]

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

clf = LogisticRegression(random_state=101)
clf.fit(train_data, label_data)
predict = clf.predict(test_data)
accuracy_score(test_label, predict)
# Score -> 32.91%

decision_tree = DecisionTreeClassifier()
decision_tree = decision_tree.fit(train_data, label_data)
dec_pred = decision_tree.predict(test_data)
accuracy_score(test_label, dec_pred)
# Score -> 35.44%

from sknn.mlp import Classifier, Layer

nn = Classifier(
    layers=[
#         Layer("Maxout", units=100, pieces=2),
        Layer("Sigmoid", units=200, dropout=0.10),
        Layer("Rectifier", units=200, dropout=0.10),
        Layer("Rectifier", units=200, dropout=0.25),
        Layer("Rectifier", units=200, dropout=0.25),
        Layer("Softmax")],
    learning_rate=0.0001,
    n_iter=200,
    batch_size=128)

nn.fit(train_data, label_data)

nn_pred = nn.predict(test_data)
accuracy_score(test_label, nn_pred)
# Score -> 22.78%






from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(lab_data.astype('int32'), nb_classes=None)

ar_99 = list()
for i in range(len(label_data)):
    if label_data[i] == 99.0:
        ar_99.append(i)

lab_data = np.copy(label_data)
for j in ar_99:
    lab_data[j] = 32

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, input_dim=275, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(33, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


model.fit(train_data, categorical_labels,
          nb_epoch=2000,
          batch_size=128)

for i in range(len(test_label)):
    if test_label[i] == 99.0:
        test_label[i] = 32.0

to_categorical(test_label, nb_classes=None).shape

model.evaluate(test_data, to_categorical(test_label.astype('int32'), nb_classes=None))



















