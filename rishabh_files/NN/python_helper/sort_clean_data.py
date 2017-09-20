import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import math

def get_sheet_field_names(excel_workbook, sheet_name) :
    # Start from row 3, as initial 2 rows contain no info
    sheet = excel_workbook.parse(sheet_name, skiprows=2, na_values=['NA'])
    # Find index of 'NOTES:' in 1st cloumn and delete all rows below it
    notes_index = sheet.loc[sheet['Field Order'] == "NOTES:"].index.tolist()[0]
    sheet = sheet.ix[1 : notes_index - 1]
    
    # select column 2,3 and 4 (Filed name, Description and Codes used)
    sheet = sheet[[1,2,3]]
    # Remove <NaN> from Field Names
    sheet = sheet.dropna(subset=[list(sheet)[0]])
    
    # Selecting Non-Yellow field names
    # Dropping <NaN> from Field Descriptions and Codes Used
    sheet_non_yellow = sheet.dropna(subset=[list(sheet)[1], list(sheet)[2]])
    
    # Selecting 'None' and Non-'None' Codes used
    sheet_code_not_none = sheet_non_yellow[sheet_non_yellow['Codes Used'] != "None"]
    sheet_code_none = sheet_non_yellow[sheet_non_yellow['Codes Used'] == "None"]
    
    # Convert all 'Field Names' to list()
    sheet_all = sheet['Field Name'].tolist()
    sheet_non_yellow = sheet_non_yellow['Field Name'].tolist()
    sheet_yellow = list(set(sheet_all) - set(sheet_non_yellow))
    sheet_code_not_none = sheet_code_not_none['Field Name'].tolist()
    sheet_code_none = sheet_code_none['Field Name'].tolist()
    
    # Output in form of list() of lists()
    output = list()
    output.append(sheet_yellow)
    output.append(sheet_non_yellow)
    output.append(sheet_all)
    output.append(sheet_code_none)
    output.append(sheet_code_not_none)
    # output = list[sheet_yellow, sheet_non_yellow, sheet_all, sheet_code_none, sheet_code_not_none]
    
    return(output)

def lowercase_32Char(list_):
    list_1 = [x.lower() for x in list_]
    list_2 = [x[0:32] for x in list_1]
    return (list_2)

def lowercase_32Char_list(field_list) :
    # Field names in CSV files are max upto 32 characters
    # and all small letters
    l = len(field_list)
    sol = list()
    
    for field in field_list:
        sol.append(lowercase_32Char(field))
    
    return(sol)

# Remove yellow fields from the data frame
def remove_yellow_fields(data_frame, yellow_field_list) :
    df_col_names = list(data_frame)
    
    drop_col = list()
    for yellow in yellow_field_list :
        if yellow in df_col_names :
            drop_col.append(yellow)
            
    df = data_frame.drop(drop_col, axis=1)
    
    if 'id' in list(df) :
        df = df.drop(['id'], axis=1)
    
    return df

def sort_dataset_state_dist_house(data_frame) :
    return (data_frame.sort(['state', 'district', 'house_no', 'house_hold_no'])).reset_index(drop=True)


def create_balanced_classes(data, classes_of_interest, var):
    columns = list(data[var].unique())
    data_filtered = data[data[var].isin(classes_of_interest)]

    # sample_no_min_class = 1e6
    # for i in classes_of_interest:
    #     if sample_no_min_class > data[data[var] == i].shape[0]:
    #         sample_no_min_class = data[data[var] == i].shape[0]
    # data_none_class = data[~data[var].isin(classes_of_interest)].sample(n = sample_no_min_class)
    # data_none_class[var] = 0.0
    # return pd.concat([data_filtered,data_none_class])
    
    return data_filtered




