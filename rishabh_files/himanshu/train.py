import tensorflow as tf
import time
import numpy as np
from evaluation import get_precision_recall_fscore


class TRAIN_NN():
	def __init__(self, tf_graph, BATCH_SIZE, EPOCHS, param_array, label_list, train_data, train_label, model_path, model_name, keep_prob,
		test_data, test_label, valid_data=None, valid_label=None, time_it=True, reset_tf_graph=True,
		save_model=True, load_model_if_saved=True):
		
		self.tf_graph = tf_graph
		self.BATCH_SIZE = BATCH_SIZE
		self.EPOCHS = EPOCHS
		self.param_array = param_array
		self.keep_prob = keep_prob
		self.label_list = label_list
		
		self.train_data = train_data
		self.test_data = test_data
		self.valid_data = valid_data

		self.train_label = train_label
		self.test_label = test_label
		self.valid_label = valid_label

		self.model_path = model_path
		self.model_name = model_path +  model_name
		
		self.time_it = time_it
		self.reset_tf_graph = reset_tf_graph
		self.save_model = save_model
		self.load_model_if_saved = load_model_if_saved

	def train(self):
		result = {}
		with tf.Session(graph=self.tf_graph) as sess:
			if self.load_model_if_saved :
				# Checking if model is saved
				if os.path.isfile(self.model_name + '.meta'):
					model_saver = tf.train.import_meta_graph(self.model_name + '.meta')
					model_saver.restore(sess, tf.train.latest_checkpoint(self.model_path + './'))
					print('Loading Saved Model')
				else :
					tf.global_variables_initializer().run()
			else :
				tf.global_variables_initializer().run()

			t1 = time.time()
			train_error, test_error = list(), list()
			
			_optimize, _error,  = self.param_array[0], self.param_array[1]
			y_pred, y_true, model_saver = self.param_array[2], self.param_array[3], self.param_array[4]
			X, y, keep_prob = self.param_array[5], self.param_array[6], self.param_array[7]

			num_batches = (self.train_data.shape[0])//self.BATCH_SIZE

			for epoch in range(self.EPOCHS):
				for i in range(num_batches - 1):
					x_batch = self.train_data[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]
					y_batch = self.train_label[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]
					
					_, train_err = sess.run([_optimize, _error], feed_dict = {X : x_batch, y : y_batch, keep_prob : self.keep_prob})
					train_error.append(train_err)

				test_err = sess.run(_error, feed_dict = {X : self.test_data, y : self.test_label, keep_prob : self.keep_prob})
				test_error.append(test_err)

			if ((self.valid_label is not None) and (self.valid_data is not None)):
				valid_err, y_pred, y_true = sess.run([_error,y_pred, y_true], feed_dict = {X : self.valid_data, y : self.valid_label, keep_prob : self.keep_prob})
				y_pred, y_true = np.array(y_pred), np.array(y_true)
			else:
				valid_err, y_pred, y_true = None, None, None
				y_pred, y_true = None, None

			train_e = sum(train_error)/len(train_error)
			test_e = sum(test_error)/len(test_error)

			prf_macro = get_precision_recall_fscore(y_true, y_pred, self.label_list, avg='macro')
			prf_micro = get_precision_recall_fscore(y_true, y_pred, self.label_list, avg='micro')
			prf_weighted = get_precision_recall_fscore(y_true, y_pred, self.label_list, avg='weighted')

			result['valid_error'], result['train_error'], result['test_error'] = valid_err, train_e, test_e
			result['macro'], result['micro'], result['weighted'] = prf_macro, prf_micro, prf_weighted

			if self.save_model :
				model_saver.save(sess, self.model_name)

			print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
			if self.time_it :
				time_taken = (time.time() - t1)/60
				result['time_taken'] = time_taken
				print('Time taken : ' + str(time_taken) + ' minutes')

			print('')
			print('Train Error : ', train_e)
			print('Test Error : ', test_e)
			print('Validation error is : ', valid_err)
			print('Validation Accuracy : ', (1.0 - float(valid_err)))
			print('MACRO - Precision, Recall and F-score : ', prf_macro)
			print('MICRO - Precision, Recall and F-score : ', prf_micro)
			print('WEIGHTED - Precision, Recall and F-score : ', prf_weighted)
			print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
			
		if self.reset_tf_graph:
			tf.reset_default_graph()

		return result

