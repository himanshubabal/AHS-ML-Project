from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_fscore_support

"""
The *_type variable specifies whether we want micro or macro variable
"""


def get_precision(true_labels, predicted_labels, precision_type = "macro"):
	return precision_score(true_labels, predicted_labels, average = precision_type)

def get_recall(true_labels, predicted_labels, precision_type = "macro"):
	return precision_score(true_labels, predicted_labels, average = precision_type)

def get_all(true_labels, predicted_labels, precision_type = "macro"):
	return precision_recall_fscore_support(true_labels, predicted_labels, average = precision_type)
