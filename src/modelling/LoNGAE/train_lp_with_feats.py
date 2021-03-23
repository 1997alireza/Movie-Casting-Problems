"""
This script trains a model based on the symmetrical autoencoder
architecture with parameter sharing. The model performs link
prediction using latent features learned from local graph topology
and available node features. The following datasets have node features:
{protein, metabolic, conflict, cora, citeseer, pubmed}

Usage: python train_lp_with_feats.py <dataset_str> <gpu_id>
"""

# import sys

# if len(sys.argv) < 3:
#     print('\nUSAGE: python %s <dataset_str> <gpu_id>' % sys.argv[0])
#     sys.exit()
gpu_id = 0

import numpy as np
import scipy.sparse as sp
from keras import backend as K
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import average_precision_score as ap_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler

from src.utils.mathematical import cosine_similarity, euclidean_distance
from .utils import generate_data, batch_data
from .utils_gcn import load_citation_data, split_citation_data
from .models.ae import autoencoder_with_node_features
import paths

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def run(adj, feats, evaluate_lp=False):
    """

    :param adj: adjacency matrix as a 2d numpy array
    :param feats: node features as a 2d numpy array
    :param evaluate_lp: if it's True, the evaluation on link prediction task is done.
    (if the evaluation is being done, the model would not be trained on all of the provided data)
    :return:
    """
    feats = MaxAbsScaler().fit_transform(feats)

    train = adj.copy()

    # TODO: for us?
    # if dataset != 'pubmed':
    #     train.setdiag(1.0)
    # else:
    #     train.setdiag(0.0)

    if evaluate_lp:
        print('\nPreparing test split...\n')
        test_inds = split_citation_data(adj)

        test_inds = np.vstack(list({tuple(row) for row in test_inds}))

        test_r = test_inds[:, 0]
        test_c = test_inds[:, 1]
        # Collect edge labels for evaluation
        # NOTE: matrix is undirected and symmetric
        labels = []
        labels.extend(np.squeeze(adj[test_r, test_c]))
        labels.extend(np.squeeze(adj[test_c, test_r]))
        # Mask test edges as missing with -1.0 values
        train[test_r, test_c] = -1.0
        train[test_c, test_r] = -1.0
        # Impute missing edges of input adj with 0.0 for good results
        adj[test_r, test_c] = 0.0
        adj[test_c, test_r] = 0.0

    # TODO: for us?
    # adj.setdiag(1.0)  # enforce self-connections

    print('\nCompiling autoencoder model...\n')

    encoder, ae = autoencoder_with_node_features(adj.shape[1], feats.shape[1])

    print(ae.summary())

    aug_adj = np.hstack((adj, feats))

    # Specify some hyperparameters
    epochs = 50
    train_batch_size = 8
    val_batch_size = 256

    print('\nFitting autoencoder model...\n')
    dummy = np.empty(shape=(aug_adj.shape[0], 1))
    y_true = dummy.copy()
    mask = dummy.copy()  # TODO: remove dummies

    training_data = generate_data(aug_adj, train, feats, y_true, mask, shuffle=True)
    b_data = batch_data(training_data, train_batch_size)
    num_iters_per_train_epoch = aug_adj.shape[0] / train_batch_size
    for e in range(epochs):
        print('\nEpoch {:d}/{:d}'.format(e + 1, epochs))
        print('Learning rate: {:6f}'.format(K.eval(ae.optimizer.lr)))
        curr_iter = 0
        train_loss = []
        for batch_aug_adj, batch_train, batch_f, dummy_y, dummy_m in b_data:
            # Each iteration/loop is a batch of train_batch_size samples
            loss = ae.train_on_batch([batch_aug_adj], [batch_train, batch_f])
            total_loss = loss[0]  # when we have multiple losses
            train_loss.append(total_loss)
            curr_iter += 1
            if curr_iter >= num_iters_per_train_epoch:
                break

        train_loss = np.asarray(train_loss)
        train_loss = np.mean(train_loss, axis=0)
        print('Avg. training loss: {:s}'.format(str(train_loss)))

        encoder.save(paths.models + 'actors_graph_ae/encoder.keras')
        ae.save(paths.models + 'actors_graph_ae/autoencoder.keras')
        print('\nTrained model is saved.')

        if evaluate_lp:
            print('\nEvaluating val set on link prediction...')
            outputs, predictions = [], []
            for step in range(int(aug_adj.shape[0] / val_batch_size + 1)):
                low = step * val_batch_size
                high = low + val_batch_size
                batch_aug_adj = aug_adj[low:high]
                if batch_aug_adj.shape[0] == 0:
                    break
                decoded_lp = ae.predict_on_batch([batch_aug_adj])[0]
                outputs.append(decoded_lp)
            decoded_lp = np.vstack(outputs)
            predictions.extend(decoded_lp[test_r, test_c])
            predictions.extend(decoded_lp[test_c, test_r])
            # print('Val AUC: {:6f}'.format(auc_score(labels, predictions)))  # continuous format is not supported
            # print('Val AP: {:6f}'.format(ap_score(labels, predictions)))  # continuous format is not supported
            print('Val CosineSim: {:6f}'.format(cosine_similarity(labels, predictions)))
            print('Val EuclideanDist: {:6f}'.format(euclidean_distance(labels, predictions)))
    print('\nAll done.')


if __name__ == '__main__':
    dataset = 'citeseer'
    print('\nLoading dataset {:s}...\n'.format(dataset))
    adj, feats, _, _, _, _, _, _ = load_citation_data(dataset)

    adj = adj.toarray()
    feats = feats.toarray()

    run(adj, feats, True)
