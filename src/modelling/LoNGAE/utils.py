import numpy as np
import networkx as nx
from scipy.io import loadmat
import scipy.sparse as sp
from itertools import combinations

np.random.seed(1982)


def generate_data(aug_adj, adj_train, feats, shuffle=True):
    zipped = list(zip(aug_adj, adj_train, feats))
    while True:  # this flag yields an infinite generator
        if shuffle:
            # print('Shuffling data')
            np.random.shuffle(zipped)
        for data in zipped:
            a, t, f = data
            yield (a, t, f)


def batch_data(data, batch_size):
    while True:  # this flag yields an infinite generator
        a, t, f = zip(*[next(data) for _ in range(batch_size)])
        a = np.vstack(a)
        t = np.vstack(t)
        f = np.vstack(f)
        yield map(np.float32, (a, t, f))


def lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5):
    from keras import backend as K
    lrate = base_lr * (1.0 - (curr_iter / float(max_iter)))**power
    K.set_value(model.optimizer.lr, lrate)
    return K.eval(model.optimizer.lr)
