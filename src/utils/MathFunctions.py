import numpy as np


def euclidean_distance(a, b):
    a_ = np.array(a)
    b_ = np.array(b)

    from numpy.linalg import norm
    return norm(a_ - b_)
