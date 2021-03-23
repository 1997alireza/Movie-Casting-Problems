from numpy import inner
from numpy.linalg import norm
import numpy as np


def cosine_similarity(a, b):
    return inner(a, b)/(norm(a)*norm(b))


def euclidean_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return norm(a - b)

