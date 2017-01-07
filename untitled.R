library(plyr)
library(compare)
require(sqldf)
library(microbenchmark)
library(matrixStats)
library(qdapTools)
library(ade4)
library(xlsx)

library(CatEncoders)

# Trying to find which cells are coloured
AHS_struct_workbook = loadWorkbook("~/Downloads/Data_structure_AHS.xlsx")

# Coloured rows have no value (<NA>) for fields of columns 'Field description' and 'code used'
AHS_struct_sheets = getSheets(AHS_struct_workbook)
AHS_struct_sheets_names = names(getSheets(AHS_struct_workbook))

##### todo -> find a way to automatically extrace files from list of sheet names


# AHS_struct_mort = read.xlsx("~/Downloads/Data_structure_AHS.xlsx", sheetName = "MORT", startRow = 3)
# # Get index of 'Field Order' == 'NOTES' and delete all rows below it
# AHS_struct_mort = AHS_struct_mort[1 : (which(AHS_struct_mort$Field.Order == "NOTES:")-1),]
# ##### todo -> use index instead of name for columns
# AHS_struct_mort_field_names = subset(AHS_struct_mort, select=c("Field.Name", "Field..Descriptions"))
# AHS_struct_mort_field_names = AHS_struct_mort_field_names[!is.na(AHS_struct_mort_field_names$Field.Name),]
# AHS_struct_mort_field_names = AHS_struct_mort_field_names[is.na(AHS_struct_mort_field_names$Field..Descriptions),]
# AHS_struct_mort_field_names = subset(AHS_struct_mort_field_names, select=c("Field.Name"))
# # Re-index
# rownames(AHS_struct_mort_field_names) = 1:nrow(AHS_struct_mort_field_names)
# # load into a vector
# AHS_struct_mort_field_names_vector = AHS_struct_mort_field_names[[1]]

func_get_sheet_field_names = function(sheet_location, sheet_name) {
	# Start from row 3, as initial 2 rows contain no info
	struct = read.xlsx(sheet_location, sheetName = sheet_name, startRow = 3)
	# Find index of 'NOTES:' in 1st cloumn and delete all rows below it
	struct = struct[1 : (which(struct[,1] == "NOTES:")-1),]

	# # select column 2 and 3 (Filed name and Description)
	# struct_field_names = subset(struct, select=c(2,3))
	# # Remove <NA> from Field Names
	# struct_field_names = struct_field_names[!is.na(struct_field_names[,1]),]
	# # Selecting Non-Yellow field names
	# struct_non_yellow = struct_field_names[!is.na(struct_field_names[,2]),]
	# # Seperate Yellow Columns --> remove non-NA Field Description
	# struct_yellow = struct_field_names[is.na(struct_field_names[,2]),]

	# # Keep only Field Name column
	# struct_yellow = subset(struct_yellow, select=c(1))
	# struct_non_yellow = subset(struct_non_yellow, select=c(1))
	# struct_all = subset(struct_field_names, select=c(1))

	# # Reindex the data
	# rownames(struct_yellow) = 1:nrow(struct_yellow)
	# rownames(struct_non_yellow) = 1:nrow(struct_non_yellow)
	# rownames(struct_all) = 1:nrow(struct_all)

	# # Load Yellow Field names in a vector
	# struct_yellow_vector = struct_yellow[[1]]
	# struct_non_yellow_vector = struct_non_yellow[[1]]
	# struct_all_vector = struct_all[[1]]

	# output = list(struct_yellow_vector, struct_non_yellow_vector, struct_all_vector)
	# ----------------------------------------------------
	# select column 2,3 and 4 (Filed name, Description and Codes used)
	struct_field_names = subset(struct, select=c(2,3,4))
	# Remove <NA> from Field Names
	struct_field_names = struct_field_names[!is.na(struct_field_names[,1]),]
	# Selecting Non-Yellow field names
	struct_non_yellow = struct_field_names[!is.na(struct_field_names[,2]),]

	# # Selecting 'None' and Non-'None' Codes used
	struct_code_not_none = struct_non_yellow[struct_non_yellow[,3] != "None",]
	struct_code_not_none = struct_code_not_none[!is.na(struct_code_not_none[,3]),]

	struct_code_none = struct_non_yellow[struct_non_yellow[,3] == "None",]
	struct_code_none = struct_code_none[!is.na(struct_code_none[,3]),]
	# struct_code_not_none = struct_field_names[struct_field_names[,3] != "None",]
	# struct_code_none = struct_field_names[struct_field_names[,3] == "None",]

	# Seperate Yellow Columns --> remove non-NA Field Description
	struct_yellow = struct_field_names[is.na(struct_field_names[,2]),]

	# Keep only Field Name column
	struct_yellow = subset(struct_yellow, select=c(1))
	struct_non_yellow = subset(struct_non_yellow, select=c(1))
	struct_all = subset(struct_field_names, select=c(1))
	struct_code_not_none = subset(struct_code_not_none, select=c(1))
	struct_code_none = subset(struct_code_none, select=c(1))

	# Reindex the data
	rownames(struct_yellow) = 1:nrow(struct_yellow)
	rownames(struct_non_yellow) = 1:nrow(struct_non_yellow)
	rownames(struct_all) = 1:nrow(struct_all)
	rownames(struct_code_not_none) = 1:nrow(struct_code_not_none)
	rownames(struct_code_none) = 1:nrow(struct_code_none)

	# Load Yellow Field names in a vector
	struct_yellow_vector = struct_yellow[[1]]
	struct_non_yellow_vector = struct_non_yellow[[1]]
	struct_all_vector = struct_all[[1]]
	struct_code_none_vector = struct_code_none[[1]]
	struct_code_not_none_vector = struct_code_not_none[[1]]

	# CODE not-NONE lies on 5th position/last position
	output = list(struct_yellow_vector, struct_non_yellow_vector, struct_all_vector, struct_code_none_vector, struct_code_not_none_vector)

	return(output)
}


lowercase_30Char_list = function(field_list) {
	# Field names in CSV files are max upto 32 characters
	# and all small letters
	l = length(field_list)
	sol = c()

	for(i in 1:l){
		lst = field_list[[i]]
		sol[[i]] = lowercase_30Char(lst)
	}
	return(sol)
}

# Turn characters of list into lowercase and limit to max 30 char
lowercase_30Char = function(list) {
	list = sapply(list, tolower)
	list = substring(list, 1, 30)
	return(list)
}

# If element in primary, but not in secondary, find it
find_uncommon_fields = function(list_primary, list_secondary) {
	lst = c()
	for(n in list_primary) {
		if(! n %in% list_secondary) {
			lst = c(lst, n)
		}
	}
	return(lst)
}

# If element in primary, but not in secondary, remove it from primary
remove_uncommon_fields = function(list_primary, list_secondary) {
	lst = find_uncommon_fields(list_primary, list_secondary)
	new_primary = list_primary[!list_primary %in% lst]
	return (new_primary)
	# Test using "' match('word', list_name) '"
}

# Find common rows in two datasets
common_rows_in_two_datasets = function(dataset_1, dataset_2) {
	return(sqldf('SELECT * FROM dataset_1 INTERSECT SELECT * FROM dataset_2'))
	# To find rows which are in 1 but not in 2
	# sqldf('SELECT * FROM a1 EXCEPT SELECT * FROM a2')
}

# Remove yellow fields from the data frame
remove_yellow_fields = function(data_frame, yellow_field_list) {
	yellow_field_list = sapply(yellow_field_list, tolower)
	df_col_names = sapply(names(data_frame), tolower)

	return (data_frame[ , !(df_col_names %in% yellow_field_list)])
}

women_field_list = lowercase_30Char_list(func_get_sheet_field_names("~/Downloads/Data_structure_AHS.xlsx", "WOMAN"))
comb_field_list = lowercase_30Char_list(func_get_sheet_field_names("~/Downloads/Data_structure_AHS.xlsx", "COMB"))
mort_field_list = lowercase_30Char_list(func_get_sheet_field_names("~/Downloads/Data_structure_AHS.xlsx", "MORT"))
wps_field_list = lowercase_30Char_list(func_get_sheet_field_names("~/Downloads/Data_structure_AHS.xlsx", "WPS"))


# Extract Common column names
common_fields_all = Reduce(intersect, list(mort_field_list[[2]], women_field_list[[2]], wps_field_list[[2]], comb_field_list[[2]]))
# ------->> 80 common fields across all datasets

# CAB = read.csv("~/Downloads/22_CAB.csv", sep="|", nrows=2)
# col_cab = colnames(CAB)


AHS_mort = read.csv("~/Downloads/22_AHS_MORT.csv", sep="|")
AHS_wps = read.csv("~/Downloads/22_AHS_WPS.csv", sep="|")

### Trying to find out if any data is common among two datasets
AHS_mort_col = lowercase_30Char(colnames(AHS_mort))
mort_field_col = mort_field_list[[2]]
mort_field_col = remove_uncommon_fields(mort_field_col, AHS_mort_col)

AHS_wps_col = lowercase_30Char(colnames(AHS_wps))
wps_field_col = wps_field_list[[2]]
wps_field_col = remove_uncommon_fields(wps_field_col, AHS_wps_col)

common_fields = Reduce(intersect, list(mort_field_col, wps_field_col))
AHS_m = subset(AHS_mort, select=c(common_fields))
AHS_w = subset(AHS_wps, select=c(common_fields))

AHS_w1 = arrange(AHS_w, district, house_no, house_hold_no)
AHS_m1 = arrange(AHS_m, district, house_no, house_hold_no)

AHS_w2 = AHS_w1[1:1000,]
AHS_m2 = AHS_m1[1:1000,]

AHS_w3 = subset(AHS_w2, select=c(1:10))
AHS_m3 = subset(AHS_m2, select=c(1:10))
# -------->> Conclusion : Only Feature names are same, data is not same
# -------->> NO DATA IS COMMON AMONG DATASETS

district_wise_dataset = function(dataset_state_wise) {
	final_list = list()
	sorted_districts = sort(unique(dataset_state_wise[,"district"]))
	# dataset_state_wise = arrange(dataset_state_wise, state, district, house_no, house_hold_no)

	i = 1
	for (dist_code in sorted_districts) {
		dist_sort = subset(dataset_state_wise, district == dist_code)
		final_list[[i]] = list(dist_code, dist_sort)
		i = i + 1
	}
	return(final_list)	
}

one_hot_df = function(new_data, one_hot_columns) {

	colnames = names(new_data)

	for(f in one_hot_columns) {
		if(f %in% colnames) {
			dummy_data = acm.disjonctif(new_data[f])
			new_data[f] = NULL
			new_data = cbind(new_data, dummy_data)
		}
	}

	return(new_data)
}

not_one_hot_colnames = function(data_frame, not_one_hot_col) {
	colnames = names(data_frame)
	i = 1
	out = list()

	for(f in not_one_hot_col) {
		if (f %in% colnames) {
			out[[i]] = f
			i = i + 1
		}
	}
	return(out)
}

one_hot_colnames = function(data_frame, not_one_hot_df_colnames) {
	return (find_uncommon_fields(colnames(data_frame), not_one_hot_df_colnames))
}

house_no_wise_dataset = function(dataset_district_wise) {
	final_list = list()
	sorted_houses = sort(unique(dataset_district_wise[,"house_no"]))
	# dataset_state_wise = arrange(dataset_state_wise, state, district, house_no, house_hold_no)

	i = 1
	for (house_code in sorted_houses) {
		house_sort = subset(dataset_district_wise, house_no == house_code)
		final_list[[i]] = list(house_code, house_sort)
		i = i + 1
	}
	return(final_list)	
}

apply_stats_on_house_no = function(data_frame) {
	# Assuming house_no is on Index-1
	# Not calculating mean, max, etc, on house_no 
	l = length(colnames(data_frame))
	# d_f = na.omit(subset(data_frame, select=c(2:l)))
	# d_f = na.omit(data_frame)
	# df = data.frame(lapply(d_f, function(cl) list(means=mean(cl,na.rm=TRUE), sds=sd(cl,na.rm=TRUE), min=min(cl,na.rm=TRUE), max=max(cl,na.rm=TRUE))))

	d_f = data_frame

	df = data.frame(lapply(d_f, function(cl) list(means=mean(cl,na.rm=TRUE), sds=sd(cl,na.rm=TRUE))))
	# Data[1:2] <- list(NULL)
	# df[2:4] = list(NULL)
	df[2] = NULL    # Deleting 'house_no.sds'
	return(df)
}

apply_stats_on_district = function(data_set_house_no_list) {
	data = data_set_house_no_list
	# data[[i]][[1]] -> will give house no
	# data[[i]][[2]] -> will give dataframe for that house

	for (i in (1:length(data))) {
		data[[i]][[2]] = apply_stats_on_house_no(data[[i]][[2]])
	}
	return (data)
}

recompile_district_dataset = function(data_set){
	l = length(data_set)
	df_list = list()

	for (i in (1:l)) {
		df_list[[i]] = data_set[[i]][[2]]
	}

	data_out = do.call("rbind", df_list)
	rownames(data_out) = 1:nrow(data_out)
	return(data_out)
}


AHS_mort_clean = remove_yellow_fields(AHS_mort, mort_field_list[[1]])
AHS_mort_clean_sorted = arrange(AHS_mort_clean, state, district, house_no, house_hold_no)

AHS_wps_clean = remove_yellow_fields(AHS_wps, wps_field_list[[1]])
AHS_wps_clean_sorted = arrange(AHS_wps_clean, state, district, house_no, house_hold_no)

#### ------ District wise test START ----
mort_22_1 = district_wise_dataset(AHS_mort_clean_sorted)[[1]][[2]]
wps_22_1 = district_wise_dataset(AHS_wps_clean_sorted)[[1]][[2]]

write.csv(mort_22_1, file="~/Downloads/22_1_MORT.csv")
write.csv(wps_22_1, file="~/Downloads/22_1_WPS.csv")

mort_22_1_onehot = one_hot_df(mort_22_1, mort_field_list[[5]])
mort_22_1_onehot_house_no = house_no_wise_dataset(mort_22_1_onehot)
mort_22_1_house_no_stats = apply_stats_on_district(mort_22_1_onehot_house_no)
mort_22_1_final = recompile_district_dataset(mort_22_1_house_no_stats)
write.csv(mort_22_1_final, file="~/Downloads/22_1_MORT_FINAL.csv")

wps_22_1_onehot = one_hot_df(wps_22_1, wps_field_list[[5]])
wps_22_1_onehot_house_no = house_no_wise_dataset(wps_22_1_onehot)
wps_22_1_house_no_stats = apply_stats_on_district(wps_22_1_onehot_house_no)
wps_22_1_final = recompile_district_dataset(wps_22_1_house_no_stats)
write.csv(wps_22_1_final, file="~/Downloads/22_1_WPS_FINAL.csv")

mort_wps_22_1_merge = merge(x = mort_22_1_final, y = wps_22_1_final, by = "house_no.means", all = TRUE)
# To Merge a lot of data frames
# Reduce(function(x, y) merge(x, y, all=TRUE), list(df1, df2, df3))
#### ------ District wise test ENDS ----

#### ------ State Wise data merging ----

mort_district_wise_dataset_list = district_wise_dataset(AHS_mort_clean_sorted)
wps_district_wise_dataset_list = district_wise_dataset(AHS_wps_clean_sorted)

#### **** Assumption -> Dataset Order -> MORT, WPS, WOMAN, COMB
make_dataset_district_wise = function(all_datasets_list){
    all_datasets_list_length = length(all_datasets_list)
    field_list = list(mort_field_list, wps_field_list, women_field_list, comb_field_list)
    dataset_clean_sorted_list = list()
    
    for (i in (1:all_datasets_list_length)) {
        dataset = all_datasets_list[[i]]

        print(dim(dataset))

        fields = field_list[[i]]

        print(length(fields[[1]]))
        
        dataset_clean = remove_yellow_fields(dataset, fields[[1]] )
        dataset_clean_sorted = arrange(dataset_clean, state, district, house_no, house_hold_no)
        # AHS_mort_clean = remove_yellow_fields(AHS_mort, mort_field_list[[1]])
        # AHS_mort_clean_sorted = arrange(AHS_mort_clean, state, district, house_no, house_hold_no)

        print(dim(dataset_clean_sorted))
        
        dataset_clean_sorted_list[[i]] = dataset_clean_sorted
    }
    
    print(length(dataset_clean_sorted_list))
    print(dim(dataset_clean_sorted_list[[1]]))
    
    
    dataset_wise_district_list = list()
    
    for (i in (1:all_datasets_list_length)) {
        district_list = unique(dataset_clean_sorted_list[[i]][,"district"])
        
        print(district_list)
        
        dataset = as.data.frame(dataset_clean_sorted_list[[i]])

        print(dim(dataset))

        fields = field_list[[i]]
        dist_final_list = list()
        
        # wps_22_1 = district_wise_dataset(AHS_wps_clean_sorted)[[1]][[2]]
        dist_dataset_list = district_wise_dataset(dataset)
        
        print(length(dist_dataset_list))
        print(dim(dist_dataset_list[[1]][[2]]))
        
        j = 1
        for (dist in district_list) {
        	print(j)
            dist_i = as.data.frame(dist_dataset_list[[j]][[2]])

            
            
            # wps_22_1_onehot = one_hot_df(wps_22_1, wps_field_list[[5]])
            # wps_22_1_onehot_house_no = house_no_wise_dataset(wps_22_1_onehot)
            # wps_22_1_house_no_stats = apply_stats_on_district(wps_22_1_onehot_house_no)
            # wps_22_1_final = recompile_district_dataset(wps_22_1_house_no_stats)
            # write.csv(wps_22_1_final, file="~/Downloads/22_1_WPS_FINAL.csv")
            
            print(dim(dist_i))
          
            # write.csv(dist_i, file="~/Downloads/test1.csv")
            dist_i_onehot = one_hot_df(dist_i, fields[[5]])
            # write.csv(dist_i_onehot, file="~/Downloads/test2.csv")
            dist_i_onehot_house_no = house_no_wise_dataset(dist_i_onehot)
            # write.csv(dist_i_onehot_house_no, file="~/Downloads/test3.csv")
            dist_i_house_no_stats = apply_stats_on_district(dist_i_onehot_house_no)
            # write.csv(dist_i_house_no_stats, file="~/Downloads/test4.csv")
            dist_i_final = recompile_district_dataset(dist_i_house_no_stats)
            # write.csv(dist_i_final, file="~/Downloads/test5.csv")
            
            csv_name = paste(c( i, "_", 22, "_", dist), collapse = "")
            file_name = paste(c("~/Downloads/", csv_name, ".csv"), collapse = "")
            print(file_name)
            write.csv(dist_i_final, file=file_name)
            
            dist_final_list[[j]] = dist_i_final
            j = j+1
        }
        dataset_wise_district_list[[i]] = dist_final_list
    }
    return(dataset_wise_district_list)
}


test_ds = head(AHS_mort_clean_sorted,2000)
test_dist_wise = district_wise_dataset(test_ds)
dim(test_dist_wise[[1]][[2]])

# Adding dtring and int
# d = paste(c("str_", 1), collapse = "")
# and -> "str_1"

merge_dataset_wise_dist_list = function(dataset_wise_district_list){

}


coln = colnames(mort_22_1)

ll = list()
i = 1
for (c in mort_field_list[[5]]){
	if (c %in% coln) {
		ll[[i]] = c
		i = i + 1
	}
}







do.call(cbind, lapply(df, function(x) table(sequence(nrow(df)), factor(x, levels = allLevels))))






