import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from utils import *
import cPickle as pickle
import argparse
# Global variables
BATCH_SIZE = 100
NUMBATCHES = 740
EPOCHS = 50

parser = argparse.ArgumentParser()
parser.add_argument("--dims", default = 302, help = "reduce dimension of the input")
parser.add_argument("--out", default = "encoded_302", help = "out file name")
args = parser.parse_args()


class Autoencoder():
	def __init__(self, dimension_arr = [280, 200, 150, 100], weigth_initialisation = tf.contrib.layers.xavier_initializer(), num_layers = 1, activation = tf.nn.relu):
		self.num_layers = num_layers
		self.activation = activation
		self.dimension_arr = dimension_arr
		self.weigth_initialisation = weigth_initialisation

	def create_normal_net(self, X):
		# creating the encoder
		prev_layer = X
		prev_layer_shape = prev_layer.get_shape().as_list()
		# print "num of layers", self.num_layers
		for layer in range(self.num_layers):
			print layer, prev_layer_shape
			with tf.variable_scope("enc_"+str(layer)):
				# print self.dimension_arr[layer], layer, self.dimension_arr
				# W = tf.get_variable("W_enc_"+str(layer),[prev_layer_shape[1], self.dimension_arr[layer]], initializer = self.weigth_initialisation)
				# b = tf.get_variable("b_enc_"+str(layer), [self.dimension_arr[layer]], initializer = self.weigth_initialisation)
				
				W = get_scope_variable("enc_"+str(layer), "W_enc_"+str(layer), [prev_layer_shape[1], self.dimension_arr[layer]])
				b = get_scope_variable("enc_"+str(layer), "b_enc_"+str(layer), [self.dimension_arr[layer]])
				
				h = self.activation(tf.matmul(prev_layer, W) + b)
				prev_layer = h
				prev_layer_shape = prev_layer.get_shape().as_list()
		print prev_layer_shape
		# creating the decoder
		for layer in range(self.num_layers-1):
			print "enteres"
			with tf.variable_scope("dec_"+str(layer)):
				# if layer == self.num_layers-1:
				# W = tf.get_variable("W_dec_"+str(layer),[prev_layer_shape[1], self.dimension_arr[self.num_layers-2 - layer]], initializer = self.weigth_initialisation)
				# b = tf.get_variable("b_dec_"+str(layer), [self.dimension_arr[self.num_layers-2 - layer]], initializer = self.weigth_initialisation)
				
				W = get_scope_variable("dec_"+str(layer), "W_dec_"+str(layer),[prev_layer_shape[1], self.dimension_arr[self.num_layers-2 - layer]])
				b = get_scope_variable("dec_"+str(layer), "b_dec_"+str(layer), [self.dimension_arr[self.num_layers-2 - layer]])	

				h = self.activation(tf.matmul(prev_layer, W) + b)
				prev_layer = h
				prev_layer_shape = prev_layer.get_shape().as_list()

		# W = tf.get_variable("W", [prev_layer_shape[1], X.get_shape().as_list()[1]], initializer = self.weigth_initialisation)
		# b = tf.get_variable("b", [X.get_shape().as_list()[1]], initializer = self.weigth_initialisation)
		
		W = get_scope_variable('', "W", [prev_layer_shape[1], X.get_shape().as_list()[1]])
		b = get_scope_variable('', "b", [X.get_shape().as_list()[1]])
		
		h = self.activation(tf.matmul(prev_layer, W) + b)

		return h, prev_layer

def get_scope_variable(scope_name, var, shape=None):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, shape, initializer=tf.contrib.layers.xavier_initializer())
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var)
    return v

def start_encoding(filename = "rs_encoded.p", red_dim = 100):
	train_data, test_data = read_data('~/AHS-ML-Project/data/')
	# print train_data.shape, test_data.shape
	X = tf.placeholder("float32", [None, 302])
	X_hat, encoded = Autoencoder([red_dim, 100]).create_normal_net(X)

	loss = tf.reduce_mean(tf.pow(X-X_hat, 2))
	optimize = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(EPOCHS):
			print "epoch no : ", epoch
			for batch_num in range(NUMBATCHES - 1):
				
				X_batch = train_data[batch_num*BATCH_SIZE:(batch_num+1)*BATCH_SIZE]
				sess.run(optimize, feed_dict = {X : X_batch})
			print "Batch loss", sess.run(loss, feed_dict = {X : X_batch})
			print "test loss", sess.run(loss, feed_dict = {X : test_data})
		out = sess.run(encoded, feed_dict = {X : np.concatenate((train_data, test_data), axis = 0)})
		pickle.dump(out, open(filename,"w"))
		# print out.shape

