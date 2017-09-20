library("caret")

require(foreign)
require(nnet)
require(ggplot2)
require(reshape2)

data = read.csv("data/22/22_AHS_COMB_diag_hotData_with_0.csv", header = TRUE, sep = ",")
y = read.csv("data/22/22_AHS_COMB_diag_col_with_0.csv", header = TRUE, sep = ",")
data_comb = cbind(data, y)
model = multinom(diagnosed_for~., data_comb, MaxNWts = 4000)
d = summary(model)
# write(d, "summary.txt")
print(d)
dput(summary(d),file="summary_lm.txt",control="all")

# print ("Hello")
