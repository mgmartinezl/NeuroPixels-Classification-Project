import numpy as np
a = [10, 8, 9, 12, 15, 16, 13, 11, 10, 9, 8, 7, 6, 12, 11, 13, 14]
a = np.asarray(a, dtype='float64')
num, bins = np.histogram(a, bins=3)

print(num)
print(bins)