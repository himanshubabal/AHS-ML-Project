import cPickle


# doubts = usual_residence

# data = ["combined", "women", "mortality", "WPS"]
def create_inp_out_data():
	data_filetype = ["combined"]
	for i in data_filetype:

		important_columns = {
			"input_columns" : [],
			"output_columns" : []
		}
		
		data = cPickle.load(open(i+"_data_all_columns.p","r"))
		
		for column in data:
			response = raw_input(column+"\n")
			if response == "i":
				important_columns["input_columns"].append(column)
			elif response == "o":
				important_columns["output_columns"].append(column)
		# print important_columns
		# print data

		cPickle.dump(important_columns, open(i+"_inp_out_cols.p","w"))

# data