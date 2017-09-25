import numpy as np

N = 20
list = [1, 2, 5, 6, 20, 1, 2, 1, 1, 19, 2, 7]

count_list = np.zeros(N)
for e in list:
    count_list[e-1] = 1

count = 0
for e in count_list:
    count+=e

print(count)






