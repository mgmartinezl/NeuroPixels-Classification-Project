import matplotlib.pyplot as plt
import numpy as np
from seaborn import distplot

a = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10]
a = np.asarray(a, dtype='float64')
num, bins = np.histogram(a, bins=15)

maxNum = max(num)
maxNum_index = np.min(np.where(num == maxNum)[0])
cutoff = np.where(num == 0)[0][-1]
halfSym = num[int(maxNum_index):len(num)]
steps = np.flipud(halfSym[1:])
fullSym = np.concatenate([steps, halfSym])
unit_percent_missing = 100 * sum(fullSym[0:cutoff]) / sum(fullSym)
unit_percent_missing = int(round(unit_percent_missing, 2))
print('unit_percent_missing', unit_percent_missing)

# num, bins = np.histogram(a, bins=15)
# maxNum_index = np.min(np.where(num == max(num))[0])
# cutoff = np.where(num == 0)[0][-1]
# unit_percent_missing = int(round((1-(sum(num) + sum(num[1:(len(num) - cutoff+1)])) / (2*sum(num)))*100, 2))
# print('unit_percent_missing', unit_percent_missing)
