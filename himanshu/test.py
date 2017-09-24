import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

data_full = pd.read_csv('/home/physics/btech/ph1140797/AHS-ML-Project/data/22/22_AHS_COMB_diag_hotData_with_0.csv')
corr_df = data_full.corr()

# np.unique((corr_df > 0.90), return_counts=True)
# p = 0.70
# np.unique((corr_df < -p), return_counts=True)[1][1] + np.unique((corr_df > p), return_counts=True)[1][1] - 294
# # Row by index name
# corr_df.loc[:'state']
# # Column by col name
# corr_df['state']
# THRESHOLD = 0.90


def get_corr(corr_df, THRESHOLD=0.90):
	res = list()
	res_2 = list()
	res_3 = list()
	row_list = list(corr_df.index.values)
	col_list = list(corr_df)
	for row in row_list:
		for col in col_list:
			if row in row_list and col in col_list:
				element = float(corr_df.loc[row][col])
				if ((element > THRESHOLD) or (element < -THRESHOLD)):
					if row != col:
						res.append([row, col, element])
						if '_' in row:
							row = row[:-4]
							if row[-1] == '_':
								row = row[:-1]
						if '_' in col:
							col = col[:-4]
							if col[-1] == '_':
								col = col[:-1]
						if row != col:
							if [row, col] not in res_2 and [col, row] not in res_2:
								res_2.append([row, col])
								res_3.append([row, col, element])
	for a in res_3:
		print(a)
	return(res, res_2, res_3)


a = get_corr(corr_df, 0.90)



