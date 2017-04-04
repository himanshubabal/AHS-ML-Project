# import numpy as np
# import cPickle as pickle

# from model import *
# from create_inp_out_column import *

# model = FFNN()
# X = tf.placeholder(tf.float32, [1000, 350])
# output = model.create_model(X)
# print output.get_shape()

import matplotlib.pyplot as plt

x = range(3)
y = [52.4, 68.1, 75.4]
plt.scatter(x,y, color = "r")
plt.xlabel("Classifier")
plt.xticks(x,["CoreNLP","SVM","XGBoost"])
plt.ylabel("Recall")
plt.title("Comparative Study of Classifiers")
plt.show()