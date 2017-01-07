library(ggplot2)

generate_plots <- function(model, class) {
	model = as.data.frame(model["coefficients"])
	model = setDT(model, keep.rownames = TRUE)[]

	model[["row_id"]] = c(1:dim(model)[1])
	print(head(model))
	# jpeg(file = “C://R//SAVEHERE//myplot.jpeg”)
	ggplot(model, aes(rn, coefficients, group=factor(row_id))) + geom_point(aes(color=factor(row_id)))

	ggsave(".\\plots\\a.pdf")
	# dev.off()

}