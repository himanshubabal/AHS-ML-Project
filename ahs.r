require(foreign)
require(nnet)
require(ggplot2)
require(reshape2)

# interpretation of p values in the p matrix/dataframe obtained
# If a p-value is less than 0.05 it is flagged with one star (*). If a p-value is less than 0.01 it is flagged with two stars (**). If a p-value is less than 0.001 it is flagged with three stars (***).


#name of the subset file used for creating the model
filename = "data/22/22_AHS_COMB_diag_hotData_with_0_subset5"

data = read.csv(paste(filename,".csv", sep = ""), header = TRUE, sep = ",") # replace the string by filename of subset file
y = read.csv("data/22/22_AHS_COMB_diag_col_with_0.csv", header = TRUE, sep = ",") #

data_comb = cbind(data, y)
model = multinom(diagnosed_for~., data_comb, MaxNWts = 4000)

d = summary(model)
z <- d$coefficients/d$standard.errors
p <- pnorm(-abs(z)) * 2
p <- (1 - pnorm(abs(z), 0, 1)) * 2
write(p, paste(filename,"_p.values", sep = ""))
# dput(summary(d),file="summary_lm.txt",control="all")

