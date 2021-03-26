import numpy as np

np.random.seed(1982)


def generate_data(aug_adj, adj_train, feats, shuffle=True):
    zipped = list(zip(aug_adj, adj_train, feats))
    while True:  # this flag yields an infinite generator
        if shuffle:
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
