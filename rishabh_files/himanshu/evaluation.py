from sklearn.metrics import precision_recall_fscore_support

# avg --> 'macro' or 'micro' or 'weighted'
def get_precision_recall_fscore(y_true, y_pred, label_list, avg='macro'):
	p, r, f, g = precision_recall_fscore_support(y_true, y_pred, average=avg, labels=label_list)
	p = float("{0:.4f}".format(p))
	r = float("{0:.4f}".format(r))
	f = float("{0:.4f}".format(f))
	return (p, r, f)