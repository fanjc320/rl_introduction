import numpy

# a = [1, 2, 3, 7, 5, 6, 7]
# list' object has no attribute 'argmax
# idx = a.argmax()
# print("idx:", idx)

# 将列表转化成数组
a_array = numpy.array(a)
# 获取最大值的索引
idx = a_array.argmax()
print("idx:", idx)