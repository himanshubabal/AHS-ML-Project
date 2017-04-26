import numpy as np
import random
import json
from FFNN import *
from Autoencoders import *
import tensorflow as tf

outputs = open("results_rs1.txt","w")
num_layers = [2, 4, 6, 8, 12, 15, 20]
hidden_layer_sizes = {
	30: (15, 20, 15, 25, 15, 20, 15, 25, 15, 20, 15, 25, 15, 20, 15, 25, 15, 20, 15),
	10: (8, 20, 8, 25, 8, 20, 8, 25, 8, 20, 8, 25, 8, 20, 8, 25, 8, 20, 8),
	50: (15, 20, 15, 25, 15, 20, 15, 25, 15, 20, 15, 25, 15, 20, 15, 25, 15, 20, 15),
	100: (80, 105, 80, 100, 70, 107, 80, 100, 70, 107, 80, 100, 70, 107, 80, 100, 70, 107)
}
activations = [tf.tanh, tf.nn.relu, tf.nn.elu, tf.nn.crelu]
lrs = [0.0001, 0.001, 0.00001, 0.000001, 0.01]
lr_decay = []
dropout = {"full" : (0.1, 0.3, 0.5, 0.8), "end_layer" : (0.4, 0.5, 0.6, 0.8, 0.9)}
full_dropout = [0, 1]
gradient_clipping = [0, 1]
autoencoded = [0, 1]
initializations = []
# epochs = [10, 100, 1000]
epochs = [100, 1000]
optimizers = [tf.train.AdadeltaOptimizer, tf.train.AdagradOptimizer, tf.train.MomentumOptimizer, tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, tf.train.FtrlOptimizer]
input_dims = [30, 10, 50, 100]
regularization = ["l1", "l2", "both", "None"]

max_perf = 100.0
idx = 202

i = 0
for j in range(1):

	# model_params = {
	# 	"layers" : random.sample(num_layers, 1)[0],
	# 	"activation" : random.sample(activations, 1)[0],
	# 	"lr" : random.sample(lrs, 1)[0],
	# 	"epoch" : random.sample(epochs, 1)[0],
	# 	"optimizer" : random.sample(optimizers, 1)[0],
	# 	"input_dim" : random.sample(input_dims, 1)[0],
	# 	"is_dropout_full" : random.sample(full_dropout, 1)[0],
	# 	"is_gradient_clipping" : random.sample(gradient_clipping, 1)[0],
	# 	"encoded_file_name" : "rs_encoded"+str(i)+".p", 
	# 	"regularization" : random.sample(regularization, 1)[0],
	# 	}

	for i_layers in num_layers :
		for i_activations in activations :
			for i_lr in lrs :
				for i_epoch in epochs :
					for i_optimizer in optimizers :
						for i_inp_dim in input_dims :
							for i_is_dropout_full in full_dropout :
								for i_is_grad_clip in gradient_clipping :
									for i_regularization in regularization :
										i += 1

										model_params = {
											"layers" : i_layers,
											"activation" : i_activations,
											"lr" : i_lr,
											"epoch" : i_epoch,
											"optimizer" : i_optimizer,
											"input_dim" : i_inp_dim,
											"is_dropout_full" : i_is_dropout_full,
											"is_gradient_clipping" : i_is_grad_clip,
											"encoded_file_name" : "rs_encoded"+str(i)+".p", 
											"regularization" : i_regularization,
														}

										if model_params["is_dropout_full"]:
											for i_dropout in dropout["full"] :
												model_params["dropout"] = i_dropout
												model_params["hidden_layer"] = hidden_layer_sizes[model_params["input_dim"]]

												print "parameter selection done. Autoencoding ..."
												start_encoding(model_params["encoded_file_name"], model_params["input_dim"])

												print "Done with Autoencoding. Making the model ..."
												output_acc = start(model_params)
												output_acc = 0.0
												if max_perf > output_acc:
													max_perf = output_acc
													idx = i
												model_params["precision"] = output_acc
												outputs.write(str(model_params)+"\n")

												print model_params
												print ("experiment no",i)

												tf.reset_default_graph()
												print('--------------------------------------')

										else:
											for i_dropout in dropout["end_layer"] :
												model_params["dropout"] = i_dropout
												model_params["hidden_layer"] = hidden_layer_sizes[model_params["input_dim"]]

												print "parameter selection done. Autoencoding ..."
												start_encoding(model_params["encoded_file_name"], model_params["input_dim"])

												print "Done with Autoencoding. Making the model ..."
												output_acc = start(model_params)
												output_acc = 0.0
												if max_perf > output_acc:
													max_perf = output_acc
													idx = i
												model_params["precision"] = output_acc
												outputs.write(str(model_params)+"\n")

												print model_params
												print ("experiment no",i)

												tf.reset_default_graph()
												print('--------------------------------------')
											
										# model_params["hidden_layer"] = hidden_layer_sizes[model_params["input_dim"]]

										# print "parameter selection done. Autoencoding ..."
										# start_encoding(model_params["encoded_file_name"], model_params["input_dim"])

										# print "Done with Autoencoding. Making the model ..."
										# output_acc = start(model_params)
										# output_acc = 0.0
										# if max_perf > output_acc:
										# 	max_perf = output_acc
										# 	idx = i
										# model_params["precision"] = output_acc
										# outputs.write(str(model_params)+"\n")
										# print model_params
										# print "experiment no",i

										# Reset the graph
										# tf.reset_default_graph()
										# print('--------------------------------------')



	# if model_params["is_dropout_full"]:
	# 	model_params["dropout"] = random.sample(dropout["full"], 1)[0]
	# else:
	# 	model_params["dropout"] = random.sample(dropout["end_layer"], 1)[0]
	# model_params["hidden_layer"] = hidden_layer_sizes[model_params["input_dim"]]

	# print "parameter selection done. Autoencoding ..."
	# start_encoding(model_params["encoded_file_name"], model_params["input_dim"])

	# print "Done with Autoencoding. Making the model ..."
	# output_acc = start(model_params)
	# # output_acc = 0.0
	# if max_perf > output_acc:
	# 	max_perf = output_acc
	# 	idx = i
	# model_params["precision"] = output_acc
	# outputs.write(str(model_params)+"\n")
	# print model_params
	# print "experiment no",i
	# # break

	# # Reset the graph
	# tf.reset_default_graph()
	# print('--------------------------------------')

outputs.close()
