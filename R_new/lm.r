# interpretation of p values in the p matrix/dataframe obtained
# If a p-value is less than 0.05 it is flagged with one star (*). If a p-value is less than 0.01 it is flagged with two stars (**). If a p-value is less than 0.001 it is flagged with three stars (***).


#name of the subset file used for creating the model
filename = "../data/22/22_AHS_COMB_diag_hotData_with_0.csv"

data = read.csv(paste(filename, sep = ""), header = TRUE, sep = ",") # replace the string by filename of subset file
y = read.csv("22_AHS_COMB_diag_col_with_0.csv", header = TRUE, sep = ",") #

# label = 1
# y[y == label] = 1
# y[y != label] = 0

data_comb = cbind(data, y)
model = lm("diagnosed_for~.", data_comb)
# sink("results.txt")
summary(model)
# sink()

