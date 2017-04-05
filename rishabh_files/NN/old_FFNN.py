keep_prob_arr = [0.01, 0.2, 0.5, 0.8]
class FFNN():
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

	  		for layer in range(1,8):
	  			with tf.variable_scope("A"+str(layer)):	
	  				W_l = tf.get_variable("W",[pre_shape[1], self.num_hidded_units[layer]], initializer = self.weigth_initialisation)
	  				b_l = tf.get_variable("b", [self.num_hidded_units[layer]], initializer = tf.random_normal_initializer())
	  				hidden_layer_l = self.activation(tf.matmul(hidden_layer_l, W_l) + b_l)
	  				hidden_layer_l_dropout = tf.nn.dropout(hidden_layer_l, keep_prob)
	  				pre_shape = hidden_layer_l_dropout.get_shape()

	  		return tf.nn.softmax(hidden_layer_l)

X = tf.placeholder(tf.float32, [None, args.inp_dims], name = "x")
y = tf.placeholder(tf.float32, [None, 7], name = "y")
keep_prob = tf.placeholder("float32")
y_ = FFNN(args.num_layers).create_model(X, [100, args.inp_dims], keep_prob)
y__ = tf.argmax(y_, 1)

cross_entropy =  tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(y_, 1e-8, 1.0)), reduction_indices=[1]))
_optimize = tf.train.AdamOptimizer(.00006).minimize(cross_entropy)
mistakes = tf.not_equal(tf.argmax(y, 1), tf.argmax(y_, 1))
_error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

train_data, train_label, test_data, test_label = read_red_data("/home/rishabh/my_stuff/a/data/", args.inp_dims)
print train_data.shape
BATCH_SIZE = 100

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	num_batches = 74000//100
	for epoch in range(600):
		prev_labels = sess.run(y_, feed_dict = {X : test_data, y : test_label, keep_prob : keep_prob_arr[2]})
		for i in range(num_batches-1):
			x_batch = train_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
			y_batch = train_label[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

			sess.run(_optimize, feed_dict = {X : x_batch, y : y_batch, keep_prob : keep_prob_arr[2]})

			if i%100 == 0:
				print "the error after ", i, " iterations is : ", sess.run(_error, feed_dict = {X : x_batch, y : y_batch, keep_prob : keep_prob_arr[2]})

		print "training error is ", sess.run(_error, feed_dict = {X : train_data, y : train_label, keep_prob : keep_prob_arr[2]})
		print "test error is ", sess.run(_error, feed_dict = {X : test_data, y : test_label, keep_prob : keep_prob_arr[2]})

		labels = sess.run(y_, feed_dict = {X : test_data, y : test_label, keep_prob : keep_prob_arr[2]})
		print np.sum(labels-prev_labels)
		











