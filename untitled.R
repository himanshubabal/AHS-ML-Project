# Extract Common column names


# Reading only first 2 rows
# We need only column names
# CSV files are too big, so loading only fto extrace col names
AHS_mort_partial = read.csv("~/Downloads/22_AHS_MORT.csv", sep="|", nrows=2)
AHS_women_partial = read.csv("~/Downloads/22_AHS_WOMEN.csv", sep="|", nrows=2)
AHS_wps_partial = read.csv("~/Downloads/22_AHS_WPS.csv", sep="|", nrows=2)
AHS_comb_partial = read.csv("~/Downloads/22_AHS_COMB.csv", sep="|", nrows=2)

col_mort = colnames(AHS_mort_partial)
col_women = colnames(AHS_women_partial)
col_wps = colnames(AHS_wps_partial)
col_comb = colnames(AHS_comb_partial)

No_of_common_fields = Reduce(intersect, list(col_mort, col_women, col_wps, col_comb))

CAB = read.csv("~/Downloads/22_CAB.csv", sep="|", nrows=2)
col_cab = colnames(CAB)


AHS_mort = read.csv("~/Downloads/22_AHS_MORT.csv", sep="|")
AHS_wps = read.csv("~/Downloads/22_AHS_WPS.csv", sep="|")

col_m = colnames(AHS_mort)
col_w = colnames(AHS_wps)

common_fields = Reduce(intersect, list(col_m, col_w))


AHS_m = subset(AHS_mort, select=c(common_fields))
AHS_w = subset(AHS_wps, select=c(common_fields))

library(plyr)
AHS_m2 = arrange(AHS_m, house_no)
AHS_w2 = arrange(AHS_w, house_no)

dim(AHS_m2)
dim(AHS_w2)

head(AHS_w2)
head(AHS_m2)


AHS_w3 = arrange(AHS_w, district, house_no)
AHS_m3 = arrange(AHS_m, district, house_no)

# Trying to find which cells are coloured
AHS_struct_workbook = loadWorkbook("~/Downloads/Data_structure_AHS.xlsx")

# Coloured rows have no value (<NA>) for fields of columns 'Field description' and 'code used'
AHS_struct_sheets = getSheets(AHS_struct_workbook)
AHS_struct_sheets_names = names(getSheets(AHS_struct_workbook))

##### todo -> find a way to automatically extrace files from list of sheet names
AHS_struct_mort = read.xlsx("~/Downloads/Data_structure_AHS.xlsx", sheetName = "MORT", startRow = 3)
# Get index of 'Field Order' == 'NOTES' and delete all rows below it
AHS_struct_mort = AHS_struct_mort[1 : (which(AHS_struct_mort$Field.Order == "NOTES:")-1),]
##### todo -> use index instead of name for columns
AHS_struct_mort_field_names = subset(AHS_struct_mort, select=c("Field.Name", "Field..Descriptions"))
AHS_struct_mort_field_names = AHS_struct_mort_field_names[!is.na(AHS_struct_mort_field_names$Field.Name),]
AHS_struct_mort_field_names = AHS_struct_mort_field_names[is.na(AHS_struct_mort_field_names$Field..Descriptions),]
AHS_struct_mort_field_names = subset(AHS_struct_mort_field_names, select=c("Field.Name"))
# This includes 3 extra rows

##### todo -> Find a way to replace 'Code Used' by actual codes instead of 0,1,2,etc


