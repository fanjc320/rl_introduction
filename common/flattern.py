# https://stackoverflow.com/questions/46862861/what-does-axes-flat-in-matplotlib-do
Let's look a minimal example, where we create some axes with plt.subplots, also see this question,

import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols=2,nrows=3, sharex=True, sharey=True)

for i, ax in enumerate(axes.flat):
    ax.scatter([i//2+1, i],[i,i//3])

plt.show()
Here, axes is a numpy array of axes,

print(type(axes))
> <type 'numpy.ndarray'>
print(axes.shape)
> (3L, 2L)
axes.flat is not a function, it's an attribute of the numpy.ndarray: numpy.ndarray.flat

ndarray.flat A 1-D iterator over the array.
This is a numpy.flatiter instance, which acts similarly to, but is not a subclass of, Python’s built-in iterator object.

Example:

import numpy as np

a = np.array([[2,3],
              [4,5],
              [6,7]])

for i in a.flat:
    print(i)
which would print the numbers 2 3 4 5 6 7.

Being an interator over the array, you can use it to loop over all the axes from the 3x2 array of axes,

for i, ax in enumerate(axes.flat):
For each iteration it would yield the next axes from that array, such that you may easily plot to all axes in a single loop.

An alternative would be to use axes.flatten(), where flatten() is method of the numpy array. Instead of an iterator, it returns a flattened version of the array:

for i, ax in enumerate(axes.flatten()):
There is no difference seen from the outside between the two. However an iterator does not actually create a new array and may hence be slightly faster (although this will never be noticable in the case of matplotlib axes objects).

flat1 = [ax for ax in axes.flat]
flat2 = axes.flatten()
print(flat1 == flat2)
> [ True  True  True  True  True  True]
Iterating a flattened version of the axes array has the advantage that you will save one loop, compared to the naive approach of iterating over rows and columns separately,

for row in axes:
    for ax in row:
        ax.scatter(...)