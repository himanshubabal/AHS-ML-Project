# import time
# import argparse
# import numpy as np
# import tensorflow as tf
# from process import *
# from process_col import *
# from tensorflow.python import debug as tf_debug

# data_path = '/home/physics/btech/ph1140797/AHS-ML-Project/data/'
# activations = [tf.tanh, tf.nn.relu, tf.nn.elu, tf.nn.crelu]
# keep_prob_arr = [0.01, 0.2, 0.7, 0.5, 0.8]


# parser = argparse.ArgumentParser(description='Pass LABLE in -label which is to be removed')
# parser.add_argument('-label', default = 0,  type=int, help = 'State for which data is to be processed -- 5, 8, 9, 10, 18, 20, 21, 22, 23')
# args = parser.parse_args()

# labels_list = [0.,  1.,  2.,  3.,  4.,  5.,  6.]
# label_to_be_removed = float(args.label)

# if label_to_be_removed not in labels_list :
# 	print('Please enter -label from [0.,  1.,  2.,  3.,  4.,  5.,  6.]')
# 	sys.exit()



# class FFNN():
# 	def __init__(self, output_shape, hidden_layers = 2, num_hidded_units = [200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300], activation = activations[0], l_r = 0.001, lr_decay = 0.01, weigth_initialisation = tf.contrib.layers.xavier_initializer()):
# 		self.hidden_layers = hidden_layers
# 		self.num_hidded_units = num_hidded_units
# 		self.activation = activation
# 		self.l_r = l_r
# 		self.lr_decay = lr_decay
# 		self.weigth_initialisation = weigth_initialisation
# 		self.output_shape = output_shape

# 	def create_model(self, data, input_shape, keep_prob):
		
# 		with tf.variable_scope("A0"):
# 			W_i = tf.get_variable("W_0",[input_shape[1], 200], initializer = self.weigth_initialisation)
# 			b_i = tf.get_variable("b_0", [200])
# 			hidden_layer_l = self.activation(tf.matmul(data, W_i) + b_i)
# 			hidden_layer_l_dropout = tf.nn.dropout(hidden_layer_l, keep_prob)	
# 			pre_shape = hidden_layer_l_dropout.get_shape()
# 		for layer in range(1,6):
# 			with tf.variable_scope("A"+str(layer)):	
# 				W_l = tf.get_variable("W",[pre_shape[1], self.num_hidded_units[layer]], initializer = self.weigth_initialisation)
# 				b_l = tf.get_variable("b", [self.num_hidded_units[layer]], initializer = tf.random_normal_initializer())
# 				hidden_layer_l = self.activation(tf.matmul(hidden_layer_l, W_l) + b_l)
# 				hidden_layer_l_dropout = tf.nn.dropout(hidden_layer_l, keep_prob)	
# 				pre_shape = hidden_layer_l_dropout.get_shape()
# 		with tf.variable_scope("Al"):
# 			W_l = tf.get_variable("W",[pre_shape[1], self.output_shape], initializer = self.weigth_initialisation)
# 			b_l = tf.get_variable("b", [self.output_shape], initializer = tf.random_normal_initializer())
# 			hidden_layer_l = tf.matmul(hidden_layer_l, W_l) + b_l
# 		return tf.nn.softmax(hidden_layer_l)

# BATCH_SIZE = 1024
# EPOCHS = 1001
# FINAL_RES = {}
# print('EPOCHS : ', EPOCHS, '  BATCH_SIZE : ', BATCH_SIZE)

# del train_data, train_label, test_data, test_label


# print('')
# print('')
# print('Now testing by removing one feature each time')
# cold_col_names_list = pd.read_csv(data_path + 'diagnosed_for_col_names.csv', low_memory=False)
# cold_col_names_list = check_unnamed(cold_col_names_list)
# cold_col_names_list = cold_col_names_list[cold_col_names_list.columns[0]].tolist()

# diagnosed_data = pd.read_csv(data_path + '22_COMB_diag_hotData.csv', low_memory=False)
# diagnosed_col = pd.read_csv(data_path + '22_COMB_diag_col.csv', low_memory=False)

# for k in range(len(cold_col_names_list)):
# 	t1 = time.time()
# 	print('-------------------------------------------------------')
# 	print("current feature to be ablated : ",cold_col_names_list[k])

# 	train_data, test_data, valid_data, label_dict, label_list = split_data_and_make_col_binary(diagnosed_data, diagnosed_col, [cold_col_names_list[k]])
	
# 	print('')
# 	print('Removing labels (making labels biary)')
# 	for label in label_list:
# 		print('Lable kept : ', label)
# 		train_label = label_dict[label]['train']
# 		test_label = label_dict[label]['test']
# 		valid_label = label_dict[label]['valid']

# 		X = tf.placeholder(tf.float32, [None, train_data.shape[1]])
# 		y = tf.placeholder(tf.float32, [None, train_label.shape[1]])
# 		keep_prob = tf.placeholder("float32")
# 		y_ = FFNN(train_label.shape[1]).create_model(X, [BATCH_SIZE, train_data.shape[1]], keep_prob)
# 		y__ = tf.argmax(y_, 1)

# 		cross_entropy =  tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(y_, 1e-8, 1.0)), reduction_indices=[1]))
# 		_optimize = tf.train.AdamOptimizer(.00006).minimize(cross_entropy)
# 		mistakes = tf.not_equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 		_error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

# 		with tf.Session() as sess:
# 			# sess.run(tf.global_variables_initializer())
# 			tf.global_variables_initializer().run()

# 			num_batches = (train_data.shape[0])//BATCH_SIZE
# 			for epoch in range(EPOCHS):

# 				# if (epoch % 100 == 0):
# 				# 	print("epoch no is ", epoch)

# 				# prev_labels = sess.run(y_, feed_dict = {X : test_data1, y : test_label, keep_prob : keep_prob_arr[2]})
# 				for i in range(num_batches - 1):
# 					x_batch = train_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
# 					y_batch = train_label[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
# 					sess.run(_optimize, feed_dict = {X : x_batch, y : y_batch, keep_prob : keep_prob_arr[2]})
# 					# sess.run(feed_dict = {X : x_batch, y : y_batch, keep_prob : keep_prob_arr[2]})

# 				# labels = sess.run(y_, feed_dict = {X : test_data1, y : test_label, keep_prob : keep_prob_arr[2]})
# 			err_k = sess.run(_error, feed_dict = {X : test_data, y : test_label, keep_prob : keep_prob_arr[2]})
# 			FINAL_RES[cold_col_names_list[k]] = err_k
# 			print('Error for feature : ' + cold_col_names_list[k] + ' : ' +str(err_k))
# 			print('Time taken : ' + str((time.time() - t1)/60) + ' minutes')
# 			print('')
		
# 		tf.reset_default_graph()
# 		print('')

# print('')
# print('-------------------------')
# print('-------------------------')
# print(FINAL_RES)
