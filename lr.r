library("caret")

setwd("C:\\Users\\Rishab\\Dropbox\\AHS-ML-Project")

source("helper.r")
source("plot.r")

max_category_size = 3
target_vars = c("sex")
uniformative_vars = c("psu_id", "identification_code", "age_code", "house_hold_no", "state_code")

data = read.csv("C:\\Users\\Rishab\\Downloads\\05.csv", header = TRUE, sep = "|")
data = data[,!(names(data) %in% uniformative_vars)]
categorical_vars = apply(as.matrix(names(data)), 1, get_categorical, data, max_category_size)
pre_process_object = preProcess(data[,(names(data) %in% categorical_vars)], method = c("center","scale"))
data[,(names(data) %in% categorical_vars)] = predict(pre_process_object, data[,(names(data) %in% categorical_vars)])



# g<-ggplot(data, aes("sex", "treatment_type"))
# g+geom_bar()
# model = train(data, "sex")
# print(data,"dfsf")
# model = lm()