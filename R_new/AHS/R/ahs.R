require(foreign)
require(nnet)
require(ggplot2)
require(reshape2)
# require(broom)

filename = "/Users/himanshubabal/Desktop/AHS-ML-Project/data/22/22_AHS_COMB_diag_hotData_with_0_subset_2_way_stars.csv"

data_csv = read.csv(paste(filename, sep = ""), header = TRUE, sep = ",")
y = read.csv("/Users/himanshubabal/Desktop/AHS-ML-Project/data/22/22_AHS_COMB_diag_col_with_0.csv", header = TRUE, sep = ",")

data_comb = cbind(data_csv, y)
model = multinom(diagnosed_for~., data_comb, MaxNWts = 4000)

d = summary(model)

z <- d$coefficients/d$standard.errors
p <- pnorm(-abs(z)) * 2
p <- (1 - pnorm(abs(z), 0, 1)) * 2
write(p, paste(filename,"_p.values", sep = ""))
# dput(summary(d),file="summary_lm.txt",control="all")

