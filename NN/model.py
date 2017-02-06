import numpy as np
import tensorflow as tf



class FFNN():
	"""docstring for FFNN"""
	def __init__(self, hidden_layers = 2, num_hidded_units = [100,100,100], activation = tf.tanh, l_r = 0.001, lr_decay = 0.01, weigth_initialisation = tf.contrib.layers.xavier_initializer()):
		self.hidden_layers = hidden_layers
		self.num_hidded_units = num_hidded_units
		self.activation = activation
		self.l_r = l_r
		self.lr_decay = lr_decay
		self.weigth_initialisation = weigth_initialisation

	def create_model(self, input_data):
		input_shape = input_data.get_shape()
		W_initial = tf.get_variable("weights_0", [input_shape[1], self.num_hidded_units[0]], initializer = self.weigth_initialisation)
		b_initial = tf.get_variable("bias_0", [self.num_hidded_units[0]])
		hidden_layer = self.activation(tf.matmul(input_data, W_initial) + b_initial)
		for layer_num in range(1, self.hidden_layers):
			hidden_shape = hidden_layer.get_shape()
			W = tf.get_variable("weights_" + str(layer_num), [hidden_shape[1], self.num_hidded_units[layer_num]], initializer = self.weigth_initialisation)
			b = tf.get_variable("bias_" + str(layer_num), [self.num_hidded_units[layer_num]], initializer = self.weigth_initialisation)
			hidden_layer = self.activation(tf.matmul(hidden_layer, W) + b)
		hidden_shape = hidden_layer.get_shape()
		W_final = tf.get_variable("weights_final", [hidden_shape[1], self.num_hidded_units[-1]], initializer = self.weigth_initialisation)
		b_final = tf.get_variable("bias_final", [self.num_hidded_units[-1]])
		output = tf.matmul(hidden_layer, W_final) + b_final
		return output

