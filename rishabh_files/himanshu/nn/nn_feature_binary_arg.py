import time
import argparse
import sys
import os
import numpy as np
import tensorflow as tf
from process import *
from process_col import *
from evaluation import *
# from tensorflow.python import debug as tf_debug

data_path = '/home/physics/btech/ph1140797/AHS-ML-Project/data/'
model_save_path = '/home/physics/btech/ph1140797/AHS-ML-Project/saved_models/'
activations = [tf.nn.relu]
keep_prob_arr = [0.01, 0.2, 0.7, 0.5, 0.8]

parser = argparse.ArgumentParser(description='Pass LABLE in -label which is to be removed')
parser.add_argument('-label', default = 0,  type=int, help = 'State for which data is to be processed -- 5, 8, 9, 10, 18, 20, 21, 22, 23')
args = parser.parse_args()

labels_list = [0.,  1.,  2.,  3.,  4.,  5.,  6.]
label_to_be_kept = float(args.label)

if label_to_be_kept not in labels_list :
	print('Please enter -label from [0.,  1.,  2.,  3.,  4.,  5.,  6.]')
	sys.exit()
print('LABEL : ', label_to_be_kept)


class FFNN():
	def __init__(self, output_shape, hidden_layers = 2, num_hidded_units = [200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300], activation = activations[0], weigth_initialisation = tf.contrib.layers.xavier_initializer()):
		self.hidden_layers = hidden_layers
		self.num_hidded_units = num_hidded_units
		self.activation = activation
		self.weigth_initialisation = weigth_initialisation
		self.output_shape = output_shape

	def create_model(self, data, input_shape, keep_prob):
		
		with tf.variable_scope("A0"):
			W_i = tf.get_variable("W_0",[input_shape[1], 200], initializer = self.weigth_initialisation)
			b_i = tf.get_variable("b_0", [200])
			hidden_layer_l = self.activation(tf.matmul(data, W_i) + b_i)
			hidden_layer_l_dropout = tf.nn.dropout(hidden_layer_l, keep_prob)	
			pre_shape = hidden_layer_l_dropout.get_shape()
		for layer in range(1,6):
			with tf.variable_scope("A"+str(layer)):	
				W_l = tf.get_variable("W",[pre_shape[1], self.num_hidded_units[layer]], initializer = self.weigth_initialisation)
				b_l = tf.get_variable("b", [self.num_hidded_units[layer]], initializer = tf.random_normal_initializer())
				hidden_layer_l = self.activation(tf.matmul(hidden_layer_l, W_l) + b_l)
				hidden_layer_l_dropout = tf.nn.dropout(hidden_layer_l, keep_prob)	
				pre_shape = hidden_layer_l_dropout.get_shape()
		with tf.variable_scope("Al"):
			W_l = tf.get_variable("W",[pre_shape[1], self.output_shape], initializer = self.weigth_initialisation)
			b_l = tf.get_variable("b", [self.output_shape], initializer = tf.random_normal_initializer())
			hidden_layer_l = tf.matmul(hidden_layer_l, W_l) + b_l
		return tf.nn.softmax(hidden_layer_l)

BATCH_SIZE = 1024
EPOCHS = 1001
FINAL_RES = {}
print('EPOCHS : ', EPOCHS, '  BATCH_SIZE : ', BATCH_SIZE)
del train_data, train_label, test_data, test_label


print('')
print('')
print('Now testing by removing one feature each time')
cold_col_names_list = pd.read_csv(data_path + 'diagnosed_for_col_names.csv', low_memory=False)
cold_col_names_list = check_unnamed(cold_col_names_list)
cold_col_names_list = cold_col_names_list[cold_col_names_list.columns[0]].tolist()

diagnosed_d = pd.read_csv(data_path + '22_COMB_diag_hotData.csv', low_memory=False)
diagnosed_c = pd.read_csv(data_path + '22_COMB_diag_col.csv', low_memory=False)

# for k in range(1):
for k in range(len(cold_col_names_list)):
	t1 = time.time()
	
	diagnosed_data = diagnosed_d.copy()
	diagnosed_col = diagnosed_c.copy()

	print('-------------------------------------------------------')
	print("current feature to be ablated : ",cold_col_names_list[k])

	print('Original : ', diagnosed_data.shape, diagnosed_col.shape)

	if (cold_col_names_list[k] == 'diagnosed_for'):
		train_data, test_data, valid_data, label_dict, label_list = split_data_and_make_col_binary(diagnosed_data, diagnosed_col, [])
	else :
		train_data, test_data, valid_data, label_dict, label_list = split_data_and_make_col_binary(diagnosed_data, diagnosed_col, [cold_col_names_list[k]])
	
	label_list = [0,1,2,3,4,5,6]
	label = label_to_be_kept
	
	train_label = label_dict[label]['train']
	test_label = label_dict[label]['test']
	valid_label = label_dict[label]['valid']

	print('Train : ', train_data.shape, train_label.shape)
	print('Test : ', test_data.shape, test_label.shape)
	print('Valid : ', valid_data.shape, valid_label.shape)

	graph_nn = tf.Graph()

	with graph_nn.as_default():
		X = tf.placeholder(tf.float32, [None, train_data.shape[1]])
		y = tf.placeholder(tf.float32, [None, train_label.shape[1]])
		keep_prob = tf.placeholder("float32")
		y_ = FFNN(train_label.shape[1]).create_model(X, [BATCH_SIZE, train_data.shape[1]], keep_prob)

		y_pred = tf.argmax(y_, 1)
		y_true = tf.argmax(y, 1)

		cross_entropy =  tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(y_, 1e-8, 1.0)), reduction_indices=[1]))
		_optimize = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
		mistakes = tf.not_equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		_error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
		model_saver = tf.train.Saver()

	model_path = model_save_path + str(label) + '/' + str(cold_col_names_list[k]) + '/'
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	model_name = model_path + '8_hidden_layers'

	with tf.Session(graph=graph_nn) as sess:
		# Checking if model is saved
		if os.path.isfile(model_name + '.meta'):
			model_saver = tf.train.import_meta_graph(model_name + '.meta')
			model_saver.restore(sess, tf.train.latest_checkpoint(model_path + './'))
			print('Loading Saved Model')
		else :
			tf.global_variables_initializer().run()

		num_batches = (train_data.shape[0])//BATCH_SIZE
		train_error = list()
		test_error = list()

		for epoch in range(EPOCHS):

			for i in range(num_batches - 1):
				x_batch = train_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
				y_batch = train_label[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
				_, train_err = sess.run([_optimize, _error], feed_dict = {X : x_batch, y : y_batch, keep_prob : keep_prob_arr[2]})
				train_error.append(train_err)

			test_err = sess.run(_error, feed_dict = {X : test_data, y : test_label, keep_prob : keep_prob_arr[2]})
			test_error.append(test_err)

		valid_err, y_pred, y_true = sess.run([_error,y_pred, y_true], feed_dict = {X : valid_data, y : valid_label, keep_prob : keep_prob_arr[2]})
		y_pred, y_true = np.array(y_pred), np.array(y_true)
		
		FINAL_RES[cold_col_names_list[k]] = valid_err
		model_saver.save(sess, model_name)
		
		print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
		print('Feature : ' + cold_col_names_list[k])
		print('Time taken : ' + str((time.time() - t1)/60) + ' minutes')
		print('')
		print('Train Error : ', sum(train_error)/len(train_error))
		print('Test Error : ', sum(test_error)/len(test_error))
		print('Full Dataset Validation error is : ', valid_err)
		print('Accuracy : ', (1.0 - float(valid_err)))
		print('MACRO - Precision, Recall and F-score : ', get_precision_recall_fscore(y_true, y_pred, label_list, avg='macro'))
		print('MICRO - Precision, Recall and F-score : ', get_precision_recall_fscore(y_true, y_pred, label_list, avg='micro'))
		print('WEIGHTED - Precision, Recall and F-score : ', get_precision_recall_fscore(y_true, y_pred, label_list, avg='weighted'))
		print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

	tf.reset_default_graph()

print('')
print('-------------------------')
print('-------------------------')
print(FINAL_RES)
