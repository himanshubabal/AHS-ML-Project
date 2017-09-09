from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
import os
import sys
import time
import pickle
import numpy as np
from ..evaluation import get_precision_recall_fscore


class SKLearn_models :
	def __init__(self, train_data, train_label, test_data, test_label, valid_data, valid_label,
				 model_path, model_name, save_model=True, time_it=True, rand_forest=False):
		self.train_data = train_data
		self.train_label = train_label
		self.test_data = test_data
		self.test_label = test_label
		self.valid_data = valid_data
		self.valid_label = valid_label
		self.save_model = save_model
		self.time_it = time_it
		self.model_path = model_path
		self.model_name = model_name
		self.rand_forest = rand_forest

	def classify(self):
		t1 = time.time()
		train_data = np.concatenate((self.train_data, self.valid_data), axis=0)
		train_label = np.concatenate((self.train_label, self.valid_label), axis=0)

		assert(train_data.shape[1] == self.test_data.shape[1])

		if not self.rand_forest:
			clf = linear_model.LogisticRegression(n_jobs=-1)
		else:
			clf = RandomForestClassifier(n_estimators=25)#, n_jobs=-1)

		clf.fit(train_data, train_label)

		if self.time_it :
			time_taken = (time.time() - t1)/60
			print('Model Training Time taken : ' + str(time_taken) + ' minutes')

		if self.save_model:
			filename = self.model_path + self.model_name + '.sav'
			if not os.path.exists(self.model_path):
				os.makedirs(model_path)
			
			pickle.dump(clf, open(filename, 'wb'))

		return(self.get_results_from_model(clf))


	def load_model(self):
		filename = self.model_path + self.model_name + '.sav'

		if os.path.isfile(filename):
			clf = pickle.load(open(filename, 'rb'))
			print('Loading Saved Model')
			return(self.get_results_from_model(clf))

		else:
			self.classify()

	def get_model(self):
		if not self.rand_forest:
			filename = self.model_path + self.model_name + '.sav'

			if os.path.isfile(filename):
				clf = pickle.load(open(filename, 'rb'))
				return clf
			else:
				self.classify()
		else:
			train_data = np.concatenate((self.train_data, self.valid_data), axis=0)
			train_label = np.concatenate((self.train_label, self.valid_label), axis=0)

			# clf1 = RandomForestClassifier()
			# clf1 = clf1.fit(train_data, train_label)
			# model = SelectFromModel(clf1, prefit=True)
			# train_data = model.transform(train_data)
			
			# print('New Shape : ', train_data.shape)
			
			clf = RandomForestClassifier(n_estimators=25)#, n_jobs=-1)
			clf.fit(train_data, train_label)
			return clf


	def get_results_from_model(self, clf):
		result = {}
		
		test_pred = clf.predict(self.test_data)
		test_error = 1.0 - float(clf.score(self.test_data, self.test_label))

		labels = np.unique(self.test_label)

		prf_macro = get_precision_recall_fscore(self.test_label, test_pred, labels, avg='macro')
		prf_micro = get_precision_recall_fscore(self.test_label, test_pred, labels, avg='micro')
		prf_weighted = get_precision_recall_fscore(self.test_label, test_pred, labels, avg='weighted')

		result['error'] = test_error
		result['macro'], result['micro'], result['weighted'] = prf_macro, prf_micro, prf_weighted
		if not self.rand_forest:
			result['slopes'] = clf.coef_
		else:
			result['slopes'] = clf.feature_importances_

		print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
		print('Test Error : ', test_error)
		print('Test Accuracy : ', (1.0 - float(test_error)))
		# if not self.rand_forest:
			# print('Slope : ', clf.coef_)
		print('MACRO - Precision, Recall and F-score : ', prf_macro)
		print('MICRO - Precision, Recall and F-score : ', prf_micro)
		print('WEIGHTED - Precision, Recall and F-score : ', prf_weighted)
		print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

		return result


