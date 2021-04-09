from numpy import inner
from numpy.linalg import norm
import numpy as np
from itertools import combinations


def cosine_similarity(a, b):
    return inner(a, b)/(norm(a)*norm(b))


def euclidean_distance(a, b):
    return norm(a - b)


def MSE(a, b):
    return np.sum((a-b)**2) / len(a)


def get_all_pairs(items):
    """
    calculates all pairs of a given list
    :param items: python built-in list
    :return: list of pairs
    """
    return list(combinations(items, 2))
