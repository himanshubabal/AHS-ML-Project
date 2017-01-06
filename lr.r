library("caret")

setwd("C:\\Users\\Rishab\\Dropbox\\AHS-ML-Project")

source("helper.r")
source("plot.r")

max_category_size = 3
target_vars = c("sex")
uniformative_vars = c("psu_id", "identification_code", "age_code", "house_hold_no", "state_code", "v54")

data1 = read.csv("C:\\Users\\Rishab\\Downloads\\05.csv", header = TRUE, sep = "|")
data = data1[,!(names(data1) %in% uniformative_vars)]
column_names = as.matrix(names(data))
categorical_vars = apply(as.matrix(names(data)), 1, get_categorical, data, max_category_size)
# column_names = as.matrix(names(data))
data = do.call("cbind", apply(column_names, 1, remove_chars, data))
pre_process_object = preProcess(data[,(names(data) %in% categorical_vars)], method = c("center"))
# s = do.call("cbind",predict(pre_process_object, data[,(names(data) %in% categorical_vars)]))
data[,(names(data) %in% categorical_vars)] = predict(pre_process_object, data[,(names(data) %in% categorical_vars)])
data[is.na(data)] = 0
a = apply(as.matrix(target_vars), 1 , analyse_feature, data)
# model = lm(as.formula("weight_in_kg~."),data)


# g<-ggplot(data, aes("sex", "treatment_type"))
# g+geom_bar()
# model = train(data, "sex")
# print(data,"dfsf")
# model = lm()