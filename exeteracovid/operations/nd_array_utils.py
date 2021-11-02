import numpy as np
from numba import njit


@njit
def resize_if_required(array, index, delta):
    if index >= len(array):
        temp_ = np.zeros(len(array) + delta, array.dtype)
        # print('temp len', len(temp_))
        temp_[:len(array)] = array
        array = temp_
        # print('resizing to', len(array))

    return array


@njit
def truncate_if_required(array, length):
    if len(array) > length:
        temp_ = np.zeros(length, array.dtype)
        temp_[:] = array[:length]
        array = temp_

    return array
