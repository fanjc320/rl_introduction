import numpy as np


if __name__ == '__main__':
    # Indexing a multi-dimensional Numpy array
    arr = np.arange(30).reshape(5,6)
    print(arr)
    coords = np.array([[0, 1], [3, 4], [3, 2]])
    print(coords)
    print(arr[coords])
    rows = coords[:,0]
    cols = coords[:,1]
    print("arr[rows, cols]:",arr[rows, cols])
    # indx = np.ravel_multi_index(coords.T, arr.shape)
    print("coords.T:", coords.T)
    indx = np.ravel_multi_index(coords.T, (5,6))
    print("indx:", indx)
    rav = arr.ravel()[indx]
    print("rav:", rav)

    #Modify multiple elements at once



