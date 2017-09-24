import operator


def sort_dict(dictonary):
	total_count = len(dictonary)
	less_then_50_count = 0
	is_error = False
	# For some cases I have accuracy,
	# while have error for others
	# So, converting all to accuracies
	# Assuming that more then 50 % of the
	# accuracies lie above 0.50, and
	# more then 50 % of the errors
	# lie below 0.50
	for key, value in dictonary.iteritems():
		if value < 0.50:
			less_then_50_count += 1
	# Counting which one is the case
	if less_then_50_count < (total_count/2):
		is_error = False
	else:
		is_error = True
	# Storing errors in new_dict		
	new_dict = {}
	if not is_error: # i.e. contains accuracies
		for key, value in dictonary.iteritems():
			new_dict[key] = "{0:.4f}".format(1.00 - value)
	else:
		for key, value in dictonary.iteritems():
			new_dict[key] = "{0:.4f}".format(value)
	# Sorted in decreasing order
	sorted_new_dict = sorted(new_dict.items(), key=operator.itemgetter(1), reverse=True)
	return sorted_new_dict

def print_sorted_dict(sorted_dict):
	for element in sorted_dict:
	    feat_name = element[0]
	    feat_error = element[1]
	    print(feat_name + ' :   ' + feat_error)

def p(dictonary):
	return(print_sorted_dict(sort_dict(dictonary)))
