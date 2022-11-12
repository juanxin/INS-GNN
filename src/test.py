import scipy.sparse as sp
import numpy as np


# two sample student t test

import numpy as np
from scipy import stats

# mean1 = 78.1
# mean2 = 73.6
#
# std1 = 0.3
# std2 = 0.1
#
# nobs1 = 1000
# nobs2 = 1000

# mean1 = 64.5
# mean2 = 59.2
#
# std1 = 0.4
# std2 = 0.4
#
# nobs1 = 1000
# nobs2 = 1000
#
# modified_std1 = np.sqrt(np.float32(nobs1)/np.float32(nobs1-1)) * std1
# modified_std2 = np.sqrt(np.float32(nobs2)/np.float32(nobs2-1)) * std2
#
# (statistic, pvalue) = stats.ttest_ind_from_stats(mean1=mean1, std1=modified_std1, nobs1=10, mean2=mean2, std2=modified_std2, nobs2=10)
#
# print("t statistic is: ", statistic)
# print("pvalue is: ", pvalue)
# adj_row = [1,2,3,4]
# adj_colume = [2,3,4,5]
# adj_value = [1,1,1,1]
#
# adj = sp.coo_matrix((adj_value, (adj_row, adj_colume)))
# adj_row_2 = adj.row
# adj_colume_2 = adj.col
#
# b=[1,2,3]
# d=[2,3,5]
# # for j in range(3):
# #     panduan = True
# #     for i in range(len(adj.row)):
# #         print(i)
# #         if adj.row[i]!=b[j] and adj.col[i]!=d[j]:
# #             continue
# #         elif adj.row[i]==b[j] and adj.col[i]==d[j]:
# #             break
# #         adj_row.append(b[j])
# #         adj_colume.append(d[j])
# #         adj_value.append(1)
# #         adj = sp.coo_matrix((adj_value, (adj_row, adj_colume)))
# for j in range(3):
#     condition = True
#     for i in range(len(adj.row)):
#         if adj.row[i]==b[j] and adj.col[i]==d[j]:
#             condition = False
#     if(condition):
#         adj_row.append(b[j])
#         adj_colume.append(d[j])
#         adj_value.append(1)
#         adj = sp.coo_matrix((adj_value, (adj_row, adj_colume)))
# print(adj)
# print(adj.todense())
a = [0.812,0.826]
print(np.std(a))