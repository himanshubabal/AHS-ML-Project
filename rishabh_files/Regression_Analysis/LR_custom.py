import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from ../plots/visualize import *
class Logisticregression(object):
	"""docstring for LogisticRegression"""
	def __init__(self, penalty = "l1"):
		self.penalty = penalty

	def train(self, train_X, train_Y, feat_column_names, output_names, model_name = "model", create_new = False):
		"""
		train_X : (num_train_samples, num_featues)
		train_Y : (num_train_samples)
		"""
		self.name = model_name
		self.feat_column_names =feat_column_names
		self.output_names = output_names

		if os.path.isdir("../Models/" + model_name ) == False or create_new == True:
			if os.path.isdir("../Models/" + model_name) == False:
				os.makedirs("../Models/" + model_name )
			model = LogisticRegression(penalty = self.penalty)
			model.fit(train_X, train_Y)
			_ = joblib.dump(model,"../Models/" + model_name+"/" + model_name+".pkl")
			self.model = model
		else:
			print "%s already exists. Using the same. set create_new to True to force create".format(model_name)
			self.model = joblib.load("../Models/%s /%s .pkl".format(model_name, model_name))

		get_important_features(self.model.coef_, self.feat_column_names,name = "../Models/" + model_name+ "/" + model_name + "hist")

	def to_categorical(self, y, nb_classes=None):
		"""Converts a class vector (integers) to binary class matrix.
		E.g. for use with categorical_crossentropy.
		# Arguments
		    y: class vector to be converted into a matrix
		        (integers from 0 to nb_classes).
		    nb_classes: total number of classes.
		# Returns
		    A binary matrix representation of the input.
		"""
		y = np.array(y, dtype='int').ravel()
		if not nb_classes:
			nb_classes = np.max(y) + 1
		n = y.shape[0]
		categorical = np.zeros((n, nb_classes))
		categorical[np.arange(n), y] = 1
		return categorical

	def test (self, test_X, test_Y, confusion_matrix = False, prc = True):
		print "go"
		if prc == True:
			model_predicition_proba = self.model.predict_proba(test_X)
			test_Y_cat = self.to_categorical(test_Y)
			plot_precision_recall_curve(test_Y_cat, model_predicition_proba, name = "../Models/" + self.name+ "/" + self.name +"prc", output_names = self.output_names)
		if confusion_matrix == True:
			model_predicition = self.model.predict(test_X)
			plot_confusion_matrix(test_Y, model_predicition, name = "../Models/" + self.name+ "/" + self.name +"cm", output_names = self.output_names)

