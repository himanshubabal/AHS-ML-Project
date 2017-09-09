"""
This file is used to create subset if r is not able to process the entire feature space
"""

import pandas as pd

#read the data
df = pd.read_csv("/Users/himanshubabal/Desktop/AHS-ML-Project/data/22/22_AHS_COMB_diag_hotData_with_0.csv", header = 0)
# df_heads = list(df.columns.values)


stars_2_way = ['district','age','rural_1.0','social_group_code_1.0','highest_qualification_1.0','highest_qualification_2.0','highest_qualification_3.0','highest_qualification_4.0','highest_qualification_5.0','highest_qualification_6.0','highest_qualification_7.0','highest_qualification_8.0','highest_qualification_9.0','occupation_status_1.0','occupation_status_2.0','occupation_status_3.0','occupation_status_4.0','occupation_status_5.0','occupation_status_6.0','occupation_status_7.0','occupation_status_8.0','occupation_status_9.0','occupation_status_10.0','occupation_status_11.0','occupation_status_13.0','occupation_status_14.0','occupation_status_15.0','occupation_status_16.0','disability_status_5.0','disability_status_7.0','injury_treatment_type_1.0','injury_treatment_type_2.0','illness_type_1.0','illness_type_4.0','treatment_source_7.0','treatment_source_9.0','treatment_source_10.0','symptoms_pertaining_illness_1.0','symptoms_pertaining_illness_2.0','symptoms_pertaining_illness_3.0','symptoms_pertaining_illness_4.0','symptoms_pertaining_illness_6.0','symptoms_pertaining_illness_7.0','symptoms_pertaining_illness_8.0','symptoms_pertaining_illness_9.0','symptoms_pertaining_illness_10.0','symptoms_pertaining_illness_11.0','symptoms_pertaining_illness_13.0','sought_medical_care_1.0','sought_medical_care_2.0','sought_medical_care_3.0','diagnosis_source_1.0','diagnosis_source_2.0','diagnosis_source_3.0','diagnosis_source_4.0','diagnosis_source_5.0','diagnosis_source_6.0','diagnosis_source_7.0','diagnosis_source_8.0','diagnosis_source_9.0','diagnosis_source_10.0','diagnosis_source_11.0','diagnosis_source_13.0','diagnosis_source_99.0','regular_treatment_1.0','regular_treatment_2.0','regular_treatment_3.0','regular_treatment_source_4.0','regular_treatment_source_9.0','regular_treatment_source_12.0','regular_treatment_source_13.0','smoke_1.0','smoke_2.0','alcohol_3.0','house_status_1.0','house_status_4.0','owner_status_1.0','owner_status_2.0','drinking_water_source_1.0','drinking_water_source_2.0','drinking_water_source_3.0','drinking_water_source_4.0','drinking_water_source_5.0','drinking_water_source_6.0','drinking_water_source_8.0','is_water_filter_1.0','water_filteration_1.0','water_filteration_2.0','water_filteration_3.0','water_filteration_4.0','water_filteration_5.0','water_filteration_6.0','water_filteration_7.0','toilet_used_1.0','toilet_used_2.0','toilet_used_5.0','toilet_used_6.0','toilet_used_7.0','toilet_used_9.0','lighting_source_1.0','cooking_fuel_1.0','cooking_fuel_2.0','cooking_fuel_3.0','cooking_fuel_4.0','cooking_fuel_5.0','cooking_fuel_6.0','cooking_fuel_7.0','kitchen_availability_3.0','kitchen_availability_4.0','is_radio_1.0','is_television_1.0','is_telephone_2.0','is_telephone_3.0','is_washing_machine_1.0','is_refrigerator_1.0','is_scooter_1.0','cart_1.0','healthscheme_1_1.0','healthscheme_1_4.0','healthscheme_1_5.0','healthscheme_1_6.0','healthscheme_2_1.0' ]

df_subset = df[stars_2_way]
df_subset.to_csv("/Users/himanshubabal/Desktop/AHS-ML-Project/data/22/22_AHS_COMB_diag_hotData_with_0_subset_2_way_stars.csv", header = True)

# subset heading that are of interest. if want to add mpre then create a
# to_keep = ["occupation_status_1.0","occupation_status_2.0","occupation_status_3.0","occupation_status_4.0","occupation_status_5.0","occupation_status_6.0","occupation_status_7.0","occupation_status_8.0","occupation_status_9.0","occupation_status_10.0","occupation_status_11.0","occupation_status_13.0","occupation_status_14.0","occupation_status_15.0","occupation_status_16.0",
# 			"healthscheme_1_1.0","healthscheme_1_2.0","healthscheme_1_3.0","healthscheme_1_4.0","healthscheme_1_5.0","healthscheme_1_6.0","healthscheme_1_7.0","healthscheme_2_1.0","healthscheme_2_2.0","healthscheme_2_3.0","healthscheme_2_4.0","healthscheme_2_5.0","healthscheme_2_6.0","healthscheme_2_7.0",
# 			"water_filteration_1.0","water_filteration_2.0","water_filteration_3.0","water_filteration_4.0","water_filteration_5.0","water_filteration_6.0","water_filteration_7.0","water_filteration_8.0",]

# to_keep1 = ["age", "highest_qualification_1.0","highest_qualification_2.0","highest_qualification_3.0","highest_qualification_4.0","highest_qualification_5.0","highest_qualification_6.0","highest_qualification_7.0","highest_qualification_8.0","highest_qualification_9.0",
# 			"currently_attending_school_1.0","currently_attending_school_2.0","currently_attending_school_3.0","treatment_source_1.0","treatment_source_2.0","treatment_source_3.0","treatment_source_4.0","treatment_source_5.0","treatment_source_6.0","treatment_source_7.0","treatment_source_8.0","treatment_source_9.0","treatment_source_10.0","treatment_source_11.0","treatment_source_13.0","treatment_source_99.0",
# 			"injury_treatment_type_1.0","injury_treatment_type_2.0","injury_treatment_type_3.0","injury_treatment_type_4.0","injury_treatment_type_5.0","injury_treatment_type_6.0","injury_treatment_type_7.0",
# 			"sought_medical_care_1.0","sought_medical_care_2.0","sought_medical_care_3.0",
# 			"chew_1.0","chew_2.0","chew_3.0","chew_4.0","chew_5.0","chew_6.0","chew_7.0","smoke_1.0","smoke_2.0","smoke_3.0","smoke_4.0","alcohol_1.0","alcohol_2.0","alcohol_3.0","alcohol_4.0",
# 			"regular_treatment_1.0","regular_treatment_2.0","regular_treatment_3.0","drinking_water_source_1.0","drinking_water_source_2.0","drinking_water_source_3.0","drinking_water_source_4.0","drinking_water_source_5.0","drinking_water_source_6.0","drinking_water_source_7.0","drinking_water_source_8.0","drinking_water_source_9.0",
# 			"is_water_filter_1.0","is_water_filter_2.0","house_structure_1.0","house_structure_2.0","house_structure_3.0","house_structure_4.0",
# 			"water_filteration_1.0","water_filteration_2.0","water_filteration_3.0","water_filteration_4.0","water_filteration_5.0","water_filteration_6.0","water_filteration_7.0","water_filteration_8.0",
# 			"healthscheme_1_1.0","healthscheme_1_2.0","healthscheme_1_3.0","healthscheme_1_4.0","healthscheme_1_5.0","healthscheme_1_6.0","healthscheme_1_7.0",
# 			"toilet_used_1.0","toilet_used_2.0","toilet_used_3.0","toilet_used_4.0","toilet_used_5.0","toilet_used_6.0","toilet_used_7.0","toilet_used_8.0","toilet_used_9.0",
# 			"is_toilet_shared_1.0","is_toilet_shared_2.0","iscoveredbyhealthscheme_1.0","iscoveredbyhealthscheme_2.0","iscoveredbyhealthscheme_3.0"]

# to_keep2 = ["is_radio_1.0","is_radio_2.0","is_television_1.0","is_television_2.0","is_computer_1.0","is_computer_2.0","is_computer_3.0","is_telephone_1.0","is_telephone_2.0","is_telephone_3.0","is_telephone_4.0","is_washing_machine_1.0","is_washing_machine_2.0","is_refrigerator_1.0","is_refrigerator_2.0","is_sewing_machine_1.0","is_sewing_machine_2.0","is_bicycle_1.0","is_bicycle_2.0","is_scooter_1.0","is_scooter_2.0","is_car_1.0","is_car_2.0","is_tractor_1.0","is_tractor_2.0","is_water_pump_1.0","is_water_pump_2.0",
# 			"lighting_source_1.0","lighting_source_2.0","lighting_source_3.0","lighting_source_4.0","lighting_source_5.0","lighting_source_6.0","cooking_fuel_1.0","cooking_fuel_2.0","cooking_fuel_3.0","cooking_fuel_4.0","cooking_fuel_5.0","cooking_fuel_6.0","cooking_fuel_7.0","cooking_fuel_8.0","cooking_fuel_9.0","kitchen_availability_1.0","kitchen_availability_2.0","kitchen_availability_3.0","kitchen_availability_4.0","kitchen_availability_5.0",] #possible correlations
#to_keep2 are the column that are possible correlations as in the mid term report

#subsets
# df_subsets = df[to_keep1 ]
# df_subsets1 = df[to_keep]
# df_subsets2 = df[to_keep1 + to_keep]
# df_subsets3 = df[to_keep2 + to_keep]
# df_subsets4 = df[to_keep2 + to_keep1]
# df_subsets5 = df[to_keep2]
# df_subsets6 = df[to_keep2 + to_keep + to_keep1]

# #writing to file
# df_subsets.to_csv("/Users/himanshubabal/Desktop/AHS-ML-Project/data/22/22_AHS_COMB_diag_hotData_with_0_subset.csv", header = True)
# df_subsets1.to_csv("/Users/himanshubabal/Desktop/AHS-ML-Project/data/22/22_AHS_COMB_diag_hotData_with_0_subset1.csv", header = True)
# df_subsets2.to_csv("/Users/himanshubabal/Desktop/AHS-ML-Project/data/22/22_AHS_COMB_diag_hotData_with_0_subset2.csv", header = True)
# df_subsets3.to_csv("/Users/himanshubabal/Desktop/AHS-ML-Project/data/22/22_AHS_COMB_diag_hotData_with_0_subset3.csv", header = True)
# df_subsets4.to_csv("/Users/himanshubabal/Desktop/AHS-ML-Project/data/22/22_AHS_COMB_diag_hotData_with_0_subset4.csv", header = True)
# df_subsets5.to_csv("/Users/himanshubabal/Desktop/AHS-ML-Project/data/22/22_AHS_COMB_diag_hotData_with_0_subset5.csv", header = True)
# df_subsets6.to_csv("/Users/himanshubabal/Desktop/AHS-ML-Project/data/22/22_AHS_COMB_diag_hotData_with_0_subset6.csv", header = True)
