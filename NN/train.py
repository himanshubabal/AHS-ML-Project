import numpy as np
import cPickle as pickle

from model import *
from create_inp_out_column import *

model = FFNN()
X = tf.placeholder(tf.float32, [1000, 350])
output = model.create_model(X)
print output.get_shape()