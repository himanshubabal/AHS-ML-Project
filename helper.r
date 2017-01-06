source("plot.r")

analyse_feature <- function(feature, data) {
	# target = data[,"sex", drop = FALSE]
	# input_data = data[,!(names(data) %in% names(target))]
	model = lm(paste(feature,"~.", sep = ""), data)
	generate_plots(model)
	prediction = predict(model)
	return(1)
}


remove_chars <- function(class, data) {
	if (is.factor(data[class][[1]]) == FALSE){
		return(data[class])
	}
	return(sapply(data[class], as.integer))
}
get_categorical <- function(class, data, max_categorical_sze) {
	# unique
	if ( nrow(unique(data[class])[class]) > max_categorical_sze ){
		return(class)
	}
}