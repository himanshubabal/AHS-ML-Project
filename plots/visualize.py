from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('GTK')
from tsne import bh_sne
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve
from collections import Counter

"""
Diagnosed_for is an containing the class name in order that the model predicts
"""
Diagnosed_for = ["None","Diabetes","Hypertension","Chronic heart disease/failure","Asthma/chronic respiratory failure","Tuberculosis","Rheumatoid arthritis /  osteoarthritis","Others (Hernia, hydrocele , peptic ulcer etc)","Others (Hernia, hydrocele , peptic ulcer etc)"]


def plot_confusion_matrix(test_labels, prediciton, var_name = "",name = "confusion_matrix", output_names = ["None","Diabetes","Hypertension","Chronic heart disease/failure","Asthma/chronic respiratory failure","Tuberculosis","Rheumatoid arthritis /  osteoarthritis","Others (Hernia, hydrocele , peptic ulcer etc)","Others (Hernia, hydrocele , peptic ulcer etc)"]):
	"""
	test_labels : output labels of the test samples (not one-hot) (num_test_samples)
	predction : prediciton of the model (not one-hot) (num_test_samples)
	"""
	print "hi", name
	# print test_labels.shape, prediciton.shape
	c_mat = confusion_matrix(test_labels, prediciton)
	df_cm = pd.DataFrame(c_mat, index = [i for i in output_names], columns = [i for i in output_names])
	# Plot a heat map for confusion matrix
	fig, ax = plt.subplots()
	fig.set_size_inches(11.7, 8.27)
	s = sns.heatmap(df_cm,  annot =True)
	ax.set_title("Confusion Matrix for {} classes of %s variable".format(len(output_names), var_name))
	ax.figure.tight_layout()
	s.figure.savefig(name + ".png")

def plot_precision_recall_curve(test_labels, y_score, var_name = "", name = "prc", colors = ["r","b","#eeefff","y","cyan","g","k","w","m"], output_names = ["None","Diabetes","Hypertension","Chronic heart disease/failure","Asthma/chronic respiratory failure","Tuberculosis","Rheumatoid arthritis /  osteoarthritis","Others (Hernia, hydrocele , peptic ulcer etc)","Others (Hernia, hydrocele , peptic ulcer etc)"]):
	"""
	y_score : obtained from the decision_function(test_X) in case of Logistic Regression or y_score is the probability distribution
	over all the output classes. its dimension is (num_samples, n_output_classes)
	test_labels : one hot vector of the test sample. its dimension is (num_samples, n_output_classes)
	"""
	# print "hi", name
	plt.figure(0)
	for i in range(y_score.shape[1]):
		precision, recall, _ = precision_recall_curve(test_labels[:,i], y_score[:,i])
		plt.plot(recall, precision, color = colors[i], label  =  output_names[i])
		plt.ylabel("Precision")
		plt.xlabel("Recall")
		plt.legend(bbox_to_anchor=(1.05, 1), loc=y_score.shape[1], borderaxespad=0.)
		plt.title("Precision Recall curve for %s".format(var_name))
	plt.savefig(name + ".png", bbox_inches="tight")
	plt.show()

def get_important_features(model_coeffs, col_names, var_name = "", name = "hist"):
	"""
	Currently this is useful for regression only
	model_coeffs : (n_classes, n_features)
	col_name : feature name
	"""
	# print "go", name

	plt.figure(1)
	coeffs = list(model_coeffs)
	important_features = []
	for weights_id in range(len(coeffs)):
		weights = coeffs[weights_id]
		a = [(i,weights[i]) for i in range(len(weights))]

		a.sort(key = lambda x: x[1])
		a = [i[0] for i in a]
		important_features += a[:5]+a[-5:]
	feat_dist = Counter(important_features)
	colnames_present = [col_names[i] for i in feat_dist.keys()]

	plt.bar(feat_dist.keys(), feat_dist.values(), color="g")
	plt.xlabel("Feature")
	plt.ylabel("Frequeny")
	# plt.xticks(feat_dist.keys(), colnames_present)
	plt.title("Feature Importance for predicting Diagnosed_for")
	plt.savefig(name + ".png", bbox_inches="tight")
	plt.show()

def visualize_data(data, routine = "PCA", dimensions = 2):
	pass








