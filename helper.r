

train <- function(data, target) {
	# target = data[,"sex", drop = FALSE]
	# input_data = data[,!(names(data) %in% names(target))]
	model = lm(target~., data)
}


get_categorical <- function(class, data, max_categorical_sze) {
	# unique
	if ( nrow(unique(data[class])[class]) > max_categorical_sze ){
		return(class)
	}
}