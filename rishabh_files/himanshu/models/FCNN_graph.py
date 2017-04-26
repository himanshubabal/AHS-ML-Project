import tensorflow as tf
from FCNN import FFNN

class TF_GRAPH():
	def __init__(self, data_shape, label_shape, BATCH_SIZE, learning_rate=0.001,
		optimizer=tf.train.AdamOptimizer):

		self.graph = tf.Graph()
		self.data_shape = data_shape
		self.label_shape = label_shape
		self.BATCH_SIZE = BATCH_SIZE
		self.optimizer = optimizer
		self.learning_rate = learning_rate

	def create_graph(self):
		with self.graph.as_default(): 
			X = tf.placeholder(tf.float32, [None, self.data_shape], name='input')
			y = tf.placeholder(tf.float32, [None, self.label_shape], name='output')
			keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

			y_ = FFNN(self.label_shape).create_model(X, [self.BATCH_SIZE, self.data_shape], keep_prob)
			y_pred = tf.argmax(y_, 1)
			y_true = tf.argmax(y, 1)

			cross_entropy =  tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(y_, 1e-8, 1.0)), reduction_indices=[1]))
			_optimize = self.optimizer(self.learning_rate).minimize(cross_entropy)
			mistakes = tf.not_equal(tf.argmax(y, 1), tf.argmax(y_, 1))
			_error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
			model_saver = tf.train.Saver()

		return (self.graph, [_optimize, _error, y_pred, y_true, model_saver, X, y, keep_prob])