import numpy as np
import tensorflow as tf
from process import *
from process_col import *
from evaluation import *

data_path = '/home/physics/btech/ph1140797/AHS-ML-Project/data/'
model_save_path = '/home/physics/btech/ph1140797/AHS-ML-Project/saved_models/'

class FFNN():
	def __init__(self, output_size, activation, hidden_layers = 2, num_hidded_units = [200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300], weigth_initialisation = tf.contrib.layers.xavier_initializer()):
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
EPOCHS = 1
print('EPOCHS : ', EPOCHS, '  BATCH_SIZE : ', BATCH_SIZE)

# activations = [tf.tanh, tf.nn.relu, tf.nn.elu, tf.nn.crelu]
activations = [tf.nn.relu]
keep_prob_arr = [0.5, 0.6, 0.7]
learning_rates = [0.005, 0.001, 0.0002]

train_data = np.load(data_path + '22_train_data.npy')
test_data = np.load(data_path + '22_test_data.npy')
train_label = np.load(data_path + '22_train_label.npy')
test_label = np.load(data_path + '22_test_label.npy')
print('Train : ', train_data.shape, train_label.shape)
print('Test : ', test_data.shape, test_label.shape)

for act in activations:
	for keep_p in keep_prob_arr:
		for lr in learning_rates:
			print('------------------------')
			print('Activation    : ', act)
			print('Keep Prob     : ', keep_p)
			print('learning Rate : ', lr)

			label_list = [0,1,2,3,4,5,6]

			X = tf.placeholder(tf.float32, [None, train_data.shape[1]])
			y = tf.placeholder(tf.float32, [None, train_label.shape[1]])
			keep_prob = tf.placeholder("float32")
			y_ = FFNN(train_label.shape[1], act).create_model(X, [BATCH_SIZE, train_data.shape[1]], keep_prob)
			
			y_pred = tf.argmax(y_, 1)
			y_true = tf.argmax(y, 1)

			cross_entropy =  tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(y_, 1e-8, 1.0)), reduction_indices=[1]))
			_optimize = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
			mistakes = tf.not_equal(tf.argmax(y, 1), tf.argmax(y_, 1))
			_error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
			model_saver = tf.train.Saver()

			print('')
			with tf.Session() as sess:
				# model_saver.restore(sess, "saved_models/CNN_New.ckpt")
				sess.run(tf.global_variables_initializer())

				num_batches = (train_data.shape[0])//BATCH_SIZE
				for epoch in range(EPOCHS):
					for i in range(num_batches-1):
						x_batch = train_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
						y_batch = train_label[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
						# _, y_pred = sess.run([_optimize, y__], feed_dict = {X : x_batch, y : y_batch, keep_prob : keep_p})
						sess.run(_optimize, feed_dict = {X : x_batch, y : y_batch, keep_prob : keep_p})

				# model_saver.save(sess, "saved_models/CNN_New.ckpt")
				
				err_1, y_pred, y_true = sess.run([_error, y_pred, y_true], feed_dict = {X : test_data, y : test_label, keep_prob : keep_p})
				y_true, y_pred = np.array(y_true), np.array(y_pred)
				assert (y_true.shape == y_pred.shape)

				print('Full Dataset error is ' + str(err_1))
				print('Accuracy : ' +  str(1.0 - float(err_1)))
				print('MACRO - Precision, Recall and F-score : ', get_precision_recall_fscore(y_true, y_pred, label_list, avg='macro'))
				print('MICRO - Precision, Recall and F-score : ', get_precision_recall_fscore(y_true, y_pred, label_list, avg='micro'))
				print('WEIGHTED - Precision, Recall and F-score : ', get_precision_recall_fscore(y_true, y_pred, label_list, avg='weighted'))
				print('')

			tf.reset_default_graph()
			print('------------------------')