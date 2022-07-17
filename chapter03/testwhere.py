import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Syntax :numpy.where(condition[, x, y])
# Parameters:
# condition : When True, yield x, otherwise yield y.
# x, y : Values from which to choose. x, y and condition need to be broadcastable to some shape.

# Returns:
# out : [ndarray or tuple of ndarrays] If both x and y are specified, the output array contains elements of x where condition is True, and elements from y elsewhere.

# If only condition is given, return the tuple condition.nonzero(), the indices where condition is True


# =======================================================
# numpy.where(condition, [x, y, ]/)
# Return elements chosen from x or y depending on condition.

# Note

# When only condition is provided, this function is a shorthand for np.asarray(condition).nonzero(). Using nonzero directly should be preferred, as it behaves correctly for subclasses. The rest of this documentation covers only the case where all three arguments are provided.

# Parameters
# conditionarray_like, bool
# Where True, yield x, otherwise yield y.

# x, yarray_like
# Values from which to choose. x, y and condition need to be broadcastable to some shape.

# Returns
# outndarray
# An array with elements from x where condition is True, and elements from y elsewhere.

# See also

# choose
# nonzero
# The function that is called when x and y are omitted

# Notes

# If all the arrays are 1-D, where is equivalent to:

# [xv if c else yv
#  for c, xv, yv in zip(condition, x, y)]
# Examples

# a = np.arange(10)
# a
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# np.where(a < 5, a, 10*a)
# array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])
# This can be used on multidimensional arrays too:

# np.where([[True, False], [True, True]],
#          [[1, 2], [3, 4]],
#          [[9, 8], [7, 6]])
# array([[1, 8],
#        [3, 4]])
# The shapes of x, y, and the condition are broadcast together:

# x, y = np.ogrid[:3, :4]
# np.where(x < y, x, 10 + y)  # both x and 10+y are broadcast
# array([[10,  0,  0,  0],
#        [10, 11,  1,  1],
#        [10, 11, 12,  2]])
# a = np.array([[0, 1, 2],
#               [0, 2, 4],
#               [0, 3, 6]])
# np.where(a < 4, a, -1)  # -1 is broadcast
# array([[ 0,  1,  2],
#        [ 0,  2, -1],
#        [ 0,  3, -1]])

def test1():
    res = np.where([[True, False], [True, True]],
         [[1, 2], [3, 4]], [[5, 6], [7, 8]])

    print('res:', res) #????

    a = np.array([[1, 2, 3], [4, 5, 6]])
  
    print("a:", a)
  
    print ('Indices of elements <4')
  
    b = np.where(a<4)
    print(b)
  
    print("Elements which are <4")
    print(a[b])# ????

def test2():
    data = np.arange(15).reshape(5, 3)
    print(data) 
 #[[ 0  1  2]
 #[ 3  4  5]
 #[ 6  7  8]
 #[ 9 10 11]
 #[12 13 14]]
    print(np.where(data>2))	
#(array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]), array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]))
    print(np.argwhere(data>2))
#[[1 0] [1 1] [1 2] [2 0] [2 1] [2 2] [3 0] [3 1] [3 2] [4 0] [4 1] [4 2]]
   

# 注意，这种情况下，也即 np.where() 用于返回断言成立时的索引，返回值的形式为 arrays of tuple，由 np.array 构成的 tuple，一般 tuple 的 len 为2（当判断的对象是多维数组时），哪怕是一维数组返回的仍是 tuple，此时tuple 的 len 为 1；

# np.where()[0] 表示行的索引，
# np.where()[1] 则表示列的索引



test2()