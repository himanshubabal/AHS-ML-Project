from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

data_path = '/home/physics/btech/ph1140797/AHS-ML-Project/data/'
columns = open(data_path + "22/22_AHS_COMB_diagnosed_for_col_names.csv","r").readlines()[1:]
columns = [col.strip() for col in columns]

data1 = pd.read_csv(data_path + "22/22_AHS_COMB_diag_hotData_with_0.csv")

# data = pd.read_csv(data_path + "22/22_AHS_COMB_diag_coldData_with_0.csv")
data = data1.drop(['state'], axis=1, errors='ignore')
hot_columns = list(data.columns.values)
data_Y = pd.read_csv(data_path + "22/22_AHS_COMB_diag_col_with_0.csv")

data = data.as_matrix()
data_Y = np.reshape(data_Y.as_matrix(),(-1,))
print data_Y.shape
# classif = mutual_info_classif(data, data_Y)
# print classif
# a =  classif.support_
# print 
imp_features = []
# for i in range(classif.shape[0]):
# 	if classif[i] == True:
# 		imp_features.append(hot_columns[i])

# print imp_features
# impo
clf = LogisticRegression()
selector = RFECV(clf, step = 1, cv = 5, n_jobs = 24)
selector.fit(data, data_Y)
for i in len(selector.support_):	
	if selector.support_[i] == True:
		imp_features.append(hot_columns[i])

print imp_features