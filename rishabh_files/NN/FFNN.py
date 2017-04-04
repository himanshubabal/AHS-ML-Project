import numpy as np
import tensorflow as tf
# from process import *
# from evaluation import *
from tensorflow.python import debug as tf_debug
from utils import *
import cPickle as pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", default = 6, help = "num of layers in the network")
parser.add_argument("--inp_dims", default = 302, help = "input dimensions")
args = parser.parse_args()

activations = [tf.tanh, tf.nn.relu, tf.nn.elu, tf.nn.crelu]
keep_prob_arr = [0.01, 0.2, 0.5, 0.8]
class FFNN():
	"""docstring for FFNN"""
	def __init__(self, tst, iter, hidden_layers = 2, num_hidded_units = [200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300], activation = activations[0], l_r = 0.001, lr_decay = 0.01, weigth_initialisation = tf.contrib.layers.xavier_initializer(), dropout_type = 0):
		self.hidden_layers = hidden_layers
		self.num_hidded_units = num_hidded_units
		self.activation = activation
		self.l_r = l_r
		self.lr_decay = lr_decay
		self.weigth_initialisation = weigth_initialisation
		self.dropout_type = dropout_type
		self.iter = iter
		self.tst = tst

	def create_model(self, data, input_shape, keep_prob):
		with tf.variable_scope("A0"):
			W_i = tf.get_variable("W_0",[input_shape[1], self.num_hidded_units[0]], initializer = self.weigth_initialisation)
			b_i = tf.get_variable("b_0", [self.num_hidded_units[0]])

			# GET self.tst and self.iter
			Y_l = tf.matmul(data, W_i)
			Y_bn, update_ema = batchnorm(Y_l, self.tst, self.iter, b_i)
			hidden_layer_l = self.activation(Y_bn)
			# hidden_layer_l = self.activation(tf.matmul(data, W_i) + b_i)

			if self.dropout_type != 0:
				hidden_layer_l = tf.nn.dropout(hidden_layer_l, keep_prob)	
			pre_shape = hidden_layer_l.get_shape()

		for layer in range(1,8):
			with tf.variable_scope("A"+str(layer)):	
				W_l = tf.get_variable("W",[pre_shape[1], self.num_hidded_units[layer]], initializer = self.weigth_initialisation)
				b_l = tf.get_variable("b", [self.num_hidded_units[layer]], initializer = tf.random_normal_initializer())

				# GET self.tst and self.iter
				Y_l = tf.matmul(hidden_layer_l, W_l)
				Y_bn, update_ema = batchnorm(Y_l, self.tst, self.iter, b_l)
				hidden_layer_l = self.activation(Y_bn)


				if self.dropout_type != 0:	
					hidden_layer_l = tf.nn.dropout(hidden_layer_l, keep_prob)	
				pre_shape = hidden_layer_l.get_shape()

		# print"ffgdfgd"
		with tf.variable_scope("Al"):
			# print "fdsfds",hidden_layer.get_shape()[1]
			W_l = tf.get_variable("W",[pre_shape[1], 7], initializer = self.weigth_initialisation)
			b_l = tf.get_variable("b", [7], initializer = tf.random_normal_initializer())
			hidden_layer_l = tf.matmul(hidden_layer_l, W_l) + b_l
			print "fdsfds",hidden_layer_l.get_shape()[1]
		return tf.nn.softmax(hidden_layer_l)
		# return hidden_layer_l

def batchnorm(Ylogits, is_test, iteration, offset):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    
    mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages

def apply_l1():
	l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.5, scope=None)
	weights = tf.trainable_variables() # all vars of your graph
	regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
	return regularization_penalty


def apply_l2():
	l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.5, scope=None)
	weights = tf.trainable_variables() # all vars of your graph
	regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
	return regularization_penalty

def start(model_params):
	X = tf.placeholder(tf.float32, [None, model_params["input_dim"]], name = "x")
	y = tf.placeholder(tf.float32, [None, 7], name = "y")
	keep_prob = tf.placeholder("float32")

	# train/test selector for batch normalisation
	tst = tf.placeholder(tf.bool)
	# training iteration
	iter = tf.placeholder(tf.int32)


	y_ = FFNN(tst, iter, model_params["layers"], model_params["hidden_layer"], model_params["activation"], model_params["lr"], dropout_type = model_params["is_dropout_full"]).create_model(X, [100, model_params["input_dim"]], keep_prob)	
	# y_ = FFNN(args.num_layers).create_model(X, [100, args.inp_dims], keep_prob)
	y__ = tf.argmax(y_, 1)
	cross_entropy =  tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(y_, 1e-8, 1.0)), reduction_indices=[1]))
	if model_params["regularization"] == "l1":
		l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.5, scope=None)
		weights = tf.trainable_variables() # all vars of your graph
		regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
		loss = cross_entropy + regularization_penalty
	elif model_params["regularization"] == "l2":
		l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.5, scope=None)
		weights = tf.trainable_variables() # all vars of your graph
		regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
		loss = cross_entropy + regularization_penalty
	elif model_params["regularization"] == "None":
		loss = cross_entropy
	else: 
		l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.5, scope=None)
		weights = tf.trainable_variables() # all vars of your graph
		regularization_penalty_l1 = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
		l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.5, scope=None)
		regularization_penalty_l2 = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
		loss = cross_entropy + regularization_penalty_l2 + regularization_penalty_l1
	# _optimize = tf.train.RMSPropOptimizer(1.0).minimize(cross_entropy)
	_optimize_fn = model_params["optimizer"](model_params["lr"])
	# _optimize_fn = tf.train.AdamOptimizer(.00006)
	if model_params["is_gradient_clipping"]:
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
		_optimize = _optimize_fn.apply_gradients(zip(grads, tvars))
	else:
		_optimize = _optimize_fn.minimize(loss)

	# _optimize = tf.train.AdamOptimizer(.00006).minimize(loss)
	# tvars = tf.trainable_variables()
	# grads, _ = tf.clip_by_global_norm(tf.gradients(cross_entropy, tvars), 1)
	# _optimize = _optimize_fn.apply_gradients(zip(grads, tvars))
	mistakes = tf.not_equal(
		tf.argmax(y, 1), tf.argmax(y_, 1))

	_error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

	train_data, train_label, test_data, test_label = read_red_data("~/AHS-ML-Project/data/", model_params["encoded_file_name"], model_params["input_dim"])
	print train_data.shape
	BATCH_SIZE = 100
	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())


		# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
		num_batches = 74000//100
		for epoch in range(model_params["epoch"]):
		# for epoch in range(1):
			prev_labels = sess.run(y_, feed_dict = {X : test_data, y : test_label, keep_prob : model_params["dropout"], tst : True, iter : epoch})
			# print list(labels),"gfdgjdgjdfio"
			for i in range(num_batches-1):

				# print i
				x_batch = train_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
				y_batch = train_label[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
				# print x_batch, y_batch
				sess.run(_optimize, feed_dict = {X : x_batch, y : y_batch, keep_prob : model_params["dropout"], tst : False, iter : i})

				if i%100 == 0:
					print "the error after ", i, " iterations is : ", sess.run(_error, feed_dict = {X : x_batch, y : y_batch, keep_prob : model_params["dropout"], tst : True, iter : i})
				# break
				# print np.isnan(sess.run(cross_entropy, feed_dict = {X : test_data, y : test_label}))
			# sess.run(_optimize, feed_dict = {X : x_batch, y : y_batch})
			print "training error is ", sess.run(_error, feed_dict = {X : train_data, y : train_label, keep_prob : model_params["dropout"], tst : False, iter : epoch})
			print "test error is ", sess.run(_error, feed_dict = {X : test_data, y : test_label, keep_prob : model_params["dropout"], tst : True, iter : epoch})
			labels = sess.run(y_, feed_dict = {X : test_data, y : test_label, keep_prob : model_params["dropout"], tst : True, iter : epoch})
			print np.sum(labels-prev_labels)
		return sess.run(_error, feed_dict = {X : test_data, y : test_label, keep_prob : model_params["dropout"], tst : True, iter : epoch})

		# print list(labels)
		# print accuracy
