import cPickle

def get_inp_out_data(df, col_dict):
	input_mat = d[col_dict["input_columns"]]
	input_mat = input_mat.as_matrix()
	output_mat = d[col_dict["output_columns"]]
	output_mat = output_mat.as_matrix()
	return input_mat, output_mat

def get_next_batch(batch_no, batch_size, input_data, output_data):
	return input_data[batch_no*batch_size : (batch_no+1)*batch_size], output_data[batch_no*batch_size : (batch_no+1)*batch_size]

	