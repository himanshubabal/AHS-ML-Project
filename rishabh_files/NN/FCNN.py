import tensorflow as tf
import variables

data_path = variables.DATA_PATH
model_save_path = variables.MODEL_SAVE_PATH


class FFNN():
	def __init__(self, output_size,
		num_hidded_units = [200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300,200,300], 
		activation = tf.nn.relu, weigth_initialisation = tf.contrib.layers.xavier_initializer()):

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



