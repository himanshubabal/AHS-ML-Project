import time
import numpy as np
import tensorflow as tf
import os
from evaluation import *

data_path = '/home/physics/btech/ph1140797/AHS-ML-Project/data/'
model_save_path = '/home/physics/btech/ph1140797/AHS-ML-Project/saved_models/'
activations = [tf.nn.relu]
keep_prob_arr = [0.01, 0.2, 0.7, 0.5, 0.8]
label_list = [0,1,2,3,4,5,6]

class FFNN():
	def __init__(self, output_size, hidden_layers = 2, num_hidded_units = [200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300], activation = activations[0], weigth_initialisation = tf.contrib.layers.xavier_initializer()):
		self.hidden_layers = hidden_layers
		self.num_hidded_units = num_hidded_units
		self.activation = activation
		self.weigth_initialisation = weigth_initialisation
		self.output_size = output_size

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
			W_l = tf.get_variable("W",[pre_shape[1], self.output_size], initializer = self.weigth_initialisation)
			b_l = tf.get_variable("b", [self.output_size], initializer = tf.random_normal_initializer())
			hidden_layer_l = tf.matmul(hidden_layer_l, W_l) + b_l
		return tf.nn.softmax(hidden_layer_l)

BATCH_SIZE = 1024
EPOCHS = 1001
FINAL_RES = {}
print('EPOCHS : ', EPOCHS, '  BATCH_SIZE : ', BATCH_SIZE)
t0 = time.time()
train_data = np.load(data_path + '_ALL_diag_train_data.npy')
train_label = np.load(data_path + '_ALL_diag_train_label.npy')
test_data = np.load(data_path + '_ALL_diag_test_data.npy')
test_label = np.load(data_path + '_ALL_diag_test_label.npy')
valid_data = np.load(data_path + '_ALL_diag_valid_data.npy')
valid_label = np.load(data_path + '_ALL_diag_valid_label.npy')
print('Read time : ', time.time() - t0)
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
	_optimize = tf.train.AdamOptimizer(.001).minimize(cross_entropy)
	mistakes = tf.not_equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	_error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
	model_saver = tf.train.Saver()


model_path = model_save_path + 'full_data_dataset' + '/'
if not os.path.exists(model_path):
	os.makedirs(model_path)
model_name = model_path + '8_hidden_layers'

print('')
print('Working with full dataset initially')
with tf.Session(graph=graph_nn) as sess:
	t2 = time.time()
	# Checking if model is saved
	if os.path.isfile(model_name + '.meta'):
		print('Saved Model found')
		model_saver = tf.train.import_meta_graph(model_name + '.meta')
		model_saver.restore(sess, tf.train.latest_checkpoint(model_path + './'))
		print('Loading Saved Model')
		# model_saver.restore(sess, model_name)
		
	else :
		print('Creating New Model')
		tf.global_variables_initializer().run()

	num_batches = (train_data.shape[0])//BATCH_SIZE
	train_error = list()
	test_error = list()

	for epoch in range(EPOCHS):
		t1 = time.time()
		for i in range(num_batches - 1):
			x_batch = train_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
			y_batch = train_label[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
			_, train_err = sess.run([_optimize, _error], feed_dict = {X : x_batch, y : y_batch, keep_prob : keep_prob_arr[2]})
			train_error.append(train_err)

		test_err = sess.run(_error, feed_dict = {X : test_data, y : test_label, keep_prob : keep_prob_arr[2]})
		print('Epoch : ', epoch, ' || Train Error : ', sum(train_error)/len(train_error), ' || Test Error : ', test_err)
		test_error.append(test_err)
		print('Time : ', time.time() - t1)

	# err_k = sess.run(_error, feed_dict = {X : test_data, y : test_label, keep_prob : keep_prob_arr[2]})
	valid_err, y_pred, y_true = sess.run([_error,y_pred, y_true], feed_dict = {X : valid_data, y : valid_label, keep_prob : keep_prob_arr[2]})
	y_pred, y_true = np.array(y_pred), np.array(y_true)
	model_saver.save(sess, model_name)
	
	print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
	print('Train Error : ', sum(train_error)/len(train_error))
	print('Test Error : ', sum(test_error)/len(test_error))
	print('Validation error is : ', valid_err)
	print('Validation Accuracy : ', (1.0 - float(valid_err)))
	print('MACRO - Precision, Recall and F-score : ', get_precision_recall_fscore(y_true, y_pred, label_list, avg='macro'))
	print('MICRO - Precision, Recall and F-score : ', get_precision_recall_fscore(y_true, y_pred, label_list, avg='micro'))
	print('WEIGHTED - Precision, Recall and F-score : ', get_precision_recall_fscore(y_true, y_pred, label_list, avg='weighted'))
	print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
	print('Time Full: ', time.time() - t2)
tf.reset_default_graph()


