
# coding: utf-8

# In[1]:

import pandas as pd


# In[131]:

import numpy as np


# In[164]:

import os


# In[236]:

import pickle


# In[167]:

data_path = "~/Documents/External_Disk_Link_WD_HDD/AHS_Data/"


# In[198]:

data_path = "/Volumes/WD HDD/AHS_Data/"


# In[199]:

AHS_struct_workbook = pd.ExcelFile(data_path + "Data_structure_AHS.xlsx")


# In[3]:

AHS_struct_workbook.sheet_names


# In[4]:

AHS_struct_sheets_names = AHS_struct_workbook.sheet_names


# In[5]:

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


# In[7]:

# Turn characters of list into lowercase and limit to max 32 char
def lowercase_32Char(list_):
    list_1 = [x.lower() for x in list_]
    list_2 = [x[0:32] for x in list_1]
    return (list_2)


# In[8]:

def lowercase_32Char_list(field_list) :
    # Field names in CSV files are max upto 32 characters
    # and all small letters
    l = len(field_list)
    sol = list()
    
    for field in field_list:
        sol.append(lowercase_32Char(field))
    
    return(sol)


# In[69]:

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


# In[88]:

def sort_dataset_state_dist_house(data_frame) :
    return (data_frame.sort(['state', 'district', 'house_no', 'house_hold_no'])).reset_index(drop=True)


# In[233]:

mort_field_list = lowercase_32Char_list(get_sheet_field_names(AHS_struct_workbook, "MORT"))
wps_field_list = lowercase_32Char_list(get_sheet_field_names(AHS_struct_workbook, "WPS"))
comb_field_list = lowercase_32Char_list(get_sheet_field_names(AHS_struct_workbook, "COMB"))
women_field_list = lowercase_32Char_list(get_sheet_field_names(AHS_struct_workbook, "WOMAN"))


# In[90]:

AHS_mort = pd.read_csv(data_path + "22_AHS_MORT.csv", sep="|")


# In[228]:

AHS_wps = pd.read_csv(data_path + "22_AHS_WPS.csv", sep="|")


# In[229]:

AHS_comb = pd.read_csv(data_path + "22_AHS_COMB.csv", sep="|")


# In[231]:

AHS_women = pd.read_csv(data_path + "22_AHS_WOMEN.csv", sep="|")


# In[91]:

AHS_mort_clean = remove_yellow_fields(AHS_mort, mort_field_list[0])


# In[92]:

AHS_mort_clean


# In[93]:

AHS_mort_clean_sorted = sort_dataset_state_dist_house(AHS_mort_clean)


# In[94]:

AHS_mort_clean_sorted


# In[95]:

def district_wise_dataset(dataframe_state) :
    sorted_state = sort_dataset_state_dist_house(dataframe_state)
    unique_dist = sorted_state['district'].unique()
    out = list()
    
    for dist in unique_dist :
        dist_sort = (sorted_state[sorted_state['district'] == dist]).reset_index(drop=True)
        out.append([dist, dist_sort])
        
    return (out)


# In[97]:

mort_dist_wise = district_wise_dataset(AHS_mort_clean_sorted)


# In[152]:

def one_hot_df(data_frame, one_hot_colnames) :
    colnames = list(data_frame)
    hot_col = list()
    
    for hot in one_hot_colnames :
        if hot in colnames :
            hot_col.append(hot)
            
    if 'district' in hot_col :
        hot_col.remove('district')
    if 'state' in hot_col :
        hot_col.remove('state')
            
    df = pd.get_dummies(data_frame, columns=hot_col)
    return (df)


# In[101]:

mort___ = one_hot_df(mort_dist_wise[0][1], mort_field_list[4])


# In[102]:

def house_no_wise_dataset(dist_data_frame) :
    unique_house = dist_data_frame['house_no'].unique()
    out = list()

    for house in unique_house :
        house_sort = (dist_data_frame[dist_data_frame['house_no'] == house]).reset_index(drop=True)
        out.append([house, house_sort])
    
    return (out)


# In[103]:

mort_house_wise = house_no_wise_dataset(mort_dist_wise[0][1])


# In[104]:

len(mort_house_wise)


# In[105]:

mort_house_wise[0][1]


# In[160]:

def recompile_district_dataset(house_level_data_list) :
    df_list = list()
    
    for i in range(len(house_level_data_list)) :
        house_df = house_level_data_list[i][1]
        df_list.append(house_df)
        
    dist_data = pd.concat(df_list)
    dist_data = dist_data.reset_index(drop=True)
    return(dist_data)


# In[147]:

def recompile_state_dataset(dist_level_data_list) :
    df_list = list()
    
    for i in range(len(dist_level_data_list)) :
        dist_df = dist_level_data_list[i]
        df_list.append(dist_df)
        
    state_data = pd.concat(df_list)
    return(state_data)


# In[107]:

dist__1 = recompile_district_dataset(house_no_wise_dataset(mort_dist_wise[0][1]))


# In[108]:

dist__1


# In[109]:

mort_dist_wise[0][1]


# In[241]:

def make_dataset_district_wise(all_datasets_list, all_field_list, state_code) :  
    
    dataset_clean_sorted_list = list()
    
    for i in range(len(all_datasets_list)) :
        dataset = all_datasets_list[i]
        field = all_field_list[i]
        
        dataset_clean = remove_yellow_fields(dataset, field[0])
        dataset_clean_sorted = sort_dataset_state_dist_house(dataset_clean)
        
        dataset_clean_sorted_list.append(dataset_clean_sorted)
        
#     file_handler = open(str(state_code) + "_clean_sorted_data_list", "wb")
#     pickle.dump(dataset_clean_sorted_list, file_handler)
#     file_handler.close()
    

    dataset_wise_district_list = list()
    
    for i in range(len(dataset_clean_sorted_list)) :
        unique_dist = dataset_clean_sorted_list[i]['district'].unique()
        field = all_field_list[i]
        dataset = dataset_clean_sorted_list[i]
        
        dist_final_list = list()
        dist_dataset_list = district_wise_dataset(dataset)
        
        j = 0
        for dist in unique_dist :
            dist_j = dist_dataset_list[j][1]
            dist_j_hot = one_hot_df(dist_j, field[4])
            dist_j_hot_house = house_no_wise_dataset(dist_j_hot)
            # Replace empty values by <NaN>
            dist_j_final = recompile_district_dataset(dist_j_hot_house)
            dist_j_final = dist_j_final.replace(r'\s+', np.nan, regex=True)

            csv_name = str(str(i) + "_" + str(state_code) + "_" + str(dist) + ".csv")

            
            file_path = str(data_path + str(state_code) + "/")
            if not os.path.exists(file_path):
                os.makedirs(file_path)
                
            file_path = file_path + str(i) + "/"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            
            file_name = file_path + csv_name
            dist_j_final.to_csv(file_name)
            
            print(file_name, "written")
            
            dist_final_list.append(dist_j_final)
            j += 1
            
        dataset_wise_district_list.append(dist_final_list)
    
    return(dataset_wise_district_list)
    


# In[232]:

csv_list = [AHS_mort, AHS_wps, AHS_women, AHS_comb]


# In[234]:

field_list = [mort_field_list, wps_field_list, women_field_list, comb_field_list]


# In[242]:

AHS_all_dist_wise = make_dataset_district_wise(csv_list, field_list, 22)


# In[240]:

AHS_mort_dist_wise_hot = make_dataset_district_wise([AHS_mort.iloc[:10000]], [mort_field_list], 22)


# In[155]:

AHS_mort_dist_wise_hot[0][10].shape


# In[161]:

state_data = recompile_state_dataset(AHS_mort_dist_wise_hot[0])


# In[162]:

state_data.shape


# In[163]:

state_data


# In[159]:

list(AHS_mort_dist_wise_hot[0][10])


# In[ ]:




# In[113]:

asd = AHS_mort.iloc[:10000]


# In[114]:

asd.shape


# In[122]:

# pd.get_dummies(asd, columns=mort_field_list[4])
one_hot_df(asd, mort_field_list[4])


# In[ ]:



