import numpy as np

arr1 = np.array([1, 2, 2, 3, 3, 4, 5, 5, 5])
arr2 = np.array([2, 3, 3, 4, 4, 5, 6, 6, 6])

arr1[:3] = arr2[:3]
print(arr1)
