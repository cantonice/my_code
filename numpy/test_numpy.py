import numpy as np

arr1 = np.array([1,2,3])
arr2 = np.array([[1,2,3],[4,5,6],[6,6,6]])
print(arr1)
print(arr2)
print(type(arr2))
print(arr2[0:1])

var = arr2.shape
print(var)
arr3 = np.array([1,2,3],ndmin = 5)
print(arr3)
print(arr3.shape)

# 数组重塑
arr4 = np.array([1,2,3,4,5,6,7,8,9])
newarr = arr4.reshape(3,3)  # 是个视图，只是形状改变了
print(newarr)

arr6 = np.array([1,2,3,4,5,6,7,8,9])
newarr6 = arr6.reshape((3,3))  # 是个视图，只是形状改变了
print(newarr6)

# 未知维数
arr5 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
newarr5 = arr5.reshape(2, 2,-1)
print(newarr5)

print(newarr5.reshape(-1))
