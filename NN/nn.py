import numpy as np
import tensorflow as tf
from process import *
from evaluation import *
from tensorflow.python import debug as tf_debug


activations = [tf.tanh, tf.nn.relu, tf.nn.elu, tf.nn.crelu]
keep_prob_arr = [0.01, 0.2, 0.5, 0.8]
class FFNN():
	"""docstring for FFNN"""
	def __init__(self, hidden_layers = 2, num_hidded_units = [200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300], activation = activations[0], l_r = 0.001, lr_decay = 0.01, weigth_initialisation = tf.contrib.layers.xavier_initializer()):
		self.hidden_layers = hidden_layers
		self.num_hidded_units = num_hidded_units
		self.activation = activation
		self.l_r = l_r
		self.lr_decay = lr_decay
		self.weigth_initialisation = weigth_initialisation
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
		print"ffgdfgd"
		with tf.variable_scope("Al"):
			# print "fdsfds",hidden_layer.get_shape()[1]
			W_l = tf.get_variable("W",[pre_shape[1], 7], initializer = self.weigth_initialisation)
			b_l = tf.get_variable("b", [7], initializer = tf.random_normal_initializer())
			hidden_layer_l = tf.matmul(hidden_layer_l, W_l) + b_l
			print "fdsfds",hidden_layer_l.get_shape()[1]
		return tf.nn.softmax(hidden_layer_l)
		# return hidden_layer_l

X = tf.placeholder(tf.float32, [None, 302], name = "x")
y = tf.placeholder(tf.float32, [None, 7], name = "y")
keep_prob = tf.placeholder("float32")
y_ = FFNN().create_model(X, [100, 302], keep_prob)
y__ = tf.argmax(y_, 1)
# print y_.get_shape()
# batch_var = tf.Variable(0)
# learning_rate = tf.train.exponential_decay(0.006, batch_var * 100, 100000, 0.95, staircase = True)
cross_entropy =  tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(y_, 1e-8, 1.0)), reduction_indices=[1]))
# l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.5, scope=None)
# weights = tf.trainable_variables() # all vars of your graph
# regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
# loss = cross_entropy + regularization_penalty
# _optimize = tf.train.RMSPropOptimizer(1.0).minimize(cross_entropy)
_optimize = tf.train.AdamOptimizer(.00006).minimize(cross_entropy)
# _optimize_fn = tf.train.AdamOptimizer(.00006)
# gvs = _optimize_fn.compute_gradients(cross_entropy)
# capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
# train_op = optimizer.apply_gradients(capped_gvs)
# _optimize = tf.train.AdamOptimizer(.00006).minimize(loss)
# tvars = tf.trainable_variables()
# grads, _ = tf.clip_by_global_norm(tf.gradients(cross_entropy, tvars), 1)
# _optimize = _optimize_fn.apply_gradients(zip(grads, tvars))
mistakes = tf.not_equal(
	tf.argmax(y, 1), tf.argmax(y_, 1))

_error = tf.reduce_mean(tf.cast(mistakes, tf.float32))


print train_data.shape
BATCH_SIZE = 100

output_file = open("dl_fi141.txt","w")
for k in range(train_data.shape[1]):
	# print train_data
	train_data1 = np.copy(train_data)
	test_data1 = np.copy(test_data)
	train_data1[:,k] = 0.0
	test_data1[:,k] = 0.0
	print "current feature to be ablated",k
	# print train_data
	# print train_data[:,2]
	# print train_data1[:,2]
	# break
	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())


		# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
		num_batches = 74000//100
		for epoch in range(600):
			print "epoch no is ", epoch
		# for epoch in range(1):
			prev_labels = sess.run(y_, feed_dict = {X : test_data1, y : test_label, keep_prob : keep_prob_arr[2]})
			# print list(labels),"gfdgjdgjdfio"
			for i in range(num_batches-1):

				# print i
				x_batch = train_data1[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
				y_batch = train_label[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
				# print x_batch, y_batch
				sess.run(_optimize, feed_dict = {X : x_batch, y : y_batch, keep_prob : keep_prob_arr[2]})

				# if i%100 == 0:
					# print "the error after ", i, " iterations is : ", sess.run(_error, feed_dict = {X : x_batch, y : y_batch, keep_prob : keep_prob_arr[2]})
				# break
				# print np.isnan(sess.run(cross_entropy, feed_dict = {X : test_data1, y : test_label}))
			# sess.run(_optimize, feed_dict = {X : x_batch, y : y_batch})
			# print "training error is ", sess.run(_error, feed_dict = {X : train_data1, y : train_label, keep_prob : keep_prob_arr[2]})
			# print "test error is ", sess.run(_error, feed_dict = {X : test_data1, y : test_label, keep_prob : keep_prob_arr[2]})
			labels = sess.run(y_, feed_dict = {X : test_data1, y : test_label, keep_prob : keep_prob_arr[2]})
			print np.sum(labels-prev_labels)
		print str(k)+","+str(sess.run(_error, feed_dict = {X : test_data1, y : test_label, keep_prob : keep_prob_arr[2]}))+"\n"
		output_file.write(str(k)+","+str(sess.run(_error, feed_dict = {X : test_data1, y : test_label, keep_prob : keep_prob_arr[2]}))+"\n")
	# # break
	# 	# print list(labels)
	# 	# print accuracy
