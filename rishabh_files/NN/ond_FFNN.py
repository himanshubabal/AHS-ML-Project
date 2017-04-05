Skip to content
This repository
Search
Pull requests
Issues
Gist
 @himanshubabal
 Sign out
 Watch 45
  Unstar 412
  Fork 174 martin-gorner/tensorflow-mnist-tutorial
 Code  Issues 3  Pull requests 1  Projects 0  Wiki  Pulse  Graphs
Branch: master Find file Copy pathtensorflow-mnist-tutorial/mnist_4.1_batchnorm_five_layers_relu.py
425eeb3  24 days ago
@meain meain Fix spacing in the network structure comment
4 contributors @martin-gorner @quarkw @yufengg @meain
RawBlameHistory     
218 lines (179 sloc)  9.98 KB
# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflowvisu
import math
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
lr = tf.placeholder(tf.float32)
# train/test selector for batch normalisation
tst = tf.placeholder(tf.bool)
# training iteration
iter = tf.placeholder(tf.int32)

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 200
M = 100
N = 60
P = 30
Q = 10

# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.ones([L])/10)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/10)
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N])/10)
W4 = tf.Variable(tf.truncated_normal([N, P], stddev=0.1))
B4 = tf.Variable(tf.ones([P])/10)
W5 = tf.Variable(tf.truncated_normal([P, Q], stddev=0.1))
B5 = tf.Variable(tf.ones([Q])/10)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

# The model
XX = tf.reshape(X, [-1, 784])

# batch norm scaling is not useful with relus
# batch norm offsets are used instead of biases

Y1l = tf.matmul(XX, W1)
Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1)
Y1 = tf.nn.relu(Y1bn)

Y2l = tf.matmul(Y1, W2)
Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2)
Y2 = tf.nn.relu(Y2bn)

Y3l = tf.matmul(Y2, W3)
Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3)
Y3 = tf.nn.relu(Y3bn)

Y4l = tf.matmul(Y3, W4)
Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
Y4 = tf.nn.relu(Y4bn)

Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1])], 0)
# to use for sigmoid
#allactivations = tf.concat([tf.reshape(Y1, [-1]), tf.reshape(Y2, [-1]), tf.reshape(Y3, [-1]), tf.reshape(Y4, [-1])], 0)
# to use for RELU
allactivations = tf.concat([tf.reduce_max(Y1, [0]), tf.reduce_max(Y2, [0]), tf.reduce_max(Y3, [0]), tf.reduce_max(Y4, [0])], 0)
alllogits = tf.concat([tf.reshape(Y1l, [-1]), tf.reshape(Y2l, [-1]), tf.reshape(Y3l, [-1]), tf.reshape(Y4l, [-1])], 0)
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
datavis = tensorflowvisu.MnistDataVis(title4="Logits", title5="Max activations across batch", histogram4colornum=2, histogram5colornum=2)


# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # learning rate decay (without batch norm)
    #max_learning_rate = 0.003
    #min_learning_rate = 0.0001
    #decay_speed = 2000
    # learning rate decay (with batch norm)
    max_learning_rate = 0.03
    min_learning_rate = 0.0001
    decay_speed = 1000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, al, ac = sess.run([accuracy, cross_entropy, I, alllogits, allactivations], {X: batch_X, Y_: batch_Y, tst: False})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        datavis.append_training_curves_data(i, a, c)
        datavis.update_image1(im)
        datavis.append_data_histograms(i, al, ac)

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], {X: mnist.test.images, Y_: mnist.test.labels, tst: True})
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False})
    sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i})

datavis.animate(training_step, iterations=10000+1, train_data_update_freq=20, test_data_update_freq=100, more_tests_at_start=True)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))