import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Lambda, add, Softmax, Activation
from keras.models import Model
from keras import optimizers
from keras import backend as K

from ..layers.custom import DenseTied
import tensorflow as tf
tf.keras.losses.MeanSquaredError

def mvn(tensor):
    """Per row mean-variance normalization."""
    epsilon = 1e-6
    mean = K.mean(tensor, axis=1, keepdims=True)
    std = K.std(tensor, axis=1, keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)
    return mvn


def mbce(y_true, y_pred):
    """ Balanced sigmoid cross-entropy loss with masking """
    mask = K.not_equal(y_true, -1.0)
    mask = K.cast(mask, dtype=np.float32)
    num_examples = K.sum(mask, axis=1)
    pos = K.cast(K.equal(y_true, 1.0), dtype=np.float32)
    num_pos = K.sum(pos, axis=None)
    neg = K.cast(K.equal(y_true, 0.0), dtype=np.float32)
    num_neg = K.sum(neg, axis=None)
    pos_ratio = 1.0 - num_pos / num_neg
    mbce = mask * tf.nn.weighted_cross_entropy_with_logits(
            targets=y_true,
            logits=y_pred,
            pos_weight=pos_ratio
    )
    mbce = K.sum(mbce, axis=1) / num_examples
    return K.mean(mbce, axis=-1)


def ce(y_true, y_pred):
    """ Sigmoid cross-entropy loss """
    return K.mean(K.binary_crossentropy(
        target=y_true,
        output=y_pred,
        from_logits=True),
        axis=-1)


def masked_ce(y_true, y_pred):
    """ Sigmoid cross-entropy loss with masking """
    mask = K.not_equal(y_true, -1.0)
    mask = K.cast(mask, dtype=np.float32)
    masked_ce = mask * K.binary_crossentropy(
        target=y_true,
        output=y_pred,
        from_logits=True)
    return K.mean(masked_ce, axis=-1)


def masked_categorical_crossentropy(y_true, y_pred):
    """ Categorical/softmax cross-entropy loss with masking """
    mask = y_true[:, -1]
    y_true = y_true[:, :-1]
    loss = K.categorical_crossentropy(target=y_true,
                                      output=y_pred,
                                      from_logits=True)
    mask = K.cast(mask, dtype=np.float32)
    loss *= mask
    return K.mean(loss, axis=-1)


def masked_mean_squared_error(y_true, y_pred):
    """ Mean Squared Error with masking """
    mask = K.not_equal(y_true, 0.0)  # ignoring edges with weight equals to zero (means there is no edge yet)
    mask = K.cast(mask, dtype=np.float32)

    mask2 = K.not_equal(y_true, -1.0)  # ignoring deleted edges based on validation set
    mask2 = K.cast(mask2, dtype=np.float32)

    err = y_true - y_pred
    masked_squared_err = K.square(err) * mask * mask2
    return K.mean(masked_squared_err, axis=-1)


def create_weighted_cosine_similarity_loss(weights):
    """ It returns a Cosine Similarity loss (= -1 * Cosine Similarity) with a balancing weight on features """
    def weighted_cosine_similarity_loss(y_true, y_pred):
        inner_prod = K.sum(y_true * y_pred * weights, axis=-1)
        y_true_norm = K.sum(y_true * y_true * weights, axis=-1)
        y_pred_norm = K.sum(y_pred * y_pred * weights, axis=-1)
        return -1 * inner_prod / (K.sqrt(y_true_norm) * K.sqrt(y_pred_norm))

    return weighted_cosine_similarity_loss


def autoencoder_with_node_features(adj_row_length, features_length, node_features_weight):
    h = adj_row_length
    w = adj_row_length + features_length

    kwargs = dict(
        use_bias=True,
        kernel_initializer='glorot_normal',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None,
        trainable=True,
    )

    data = Input(shape=(w,), dtype=np.float32, name='data')

    noisy_data = Dropout(rate=0.5, name='drop1')(data)

       
    ### First set of encoding transformation ###
    encoded = Dense(256, activation='relu',
            name='encoded1', **kwargs)(noisy_data)
    encoded = Lambda(mvn, name='mvn1')(encoded)

    ### Second set of encoding transformation ###
    encoded = Dense(128, activation='relu',
            name='encoded2', **kwargs)(encoded)
    encoded = Lambda(mvn, name='mvn2')(encoded)
    encoded = Dropout(rate=0.5, name='drop2')(encoded)

    # the encoder model maps an input to its encoded representation
    encoder = Model([data], encoded)
    encoded1 = encoder.get_layer('encoded1')
    encoded2 = encoder.get_layer('encoded2')

    ### First set of decoding transformation ###
    decoded = DenseTied(256, tie_to=encoded2, transpose=True,
            activation='relu', name='decoded2')(encoded)
    decoded = Lambda(mvn, name='mvn3')(decoded)

    ### Second set of decoding transformation - reconstruction ###
    decoded = DenseTied(w, tie_to=encoded1, transpose=True,
            activation='linear', name='decoded1')(decoded)

    # output related to node features
    decoded_feats_logits = Lambda(lambda x: x[:, h:],
                        name='decoded_feats_logits')(decoded)

    decoded_feats = Softmax(name='decoded_feats')(decoded_feats_logits)

    # output related to adjacency
    decoded_adj_logits = Lambda(lambda x: x[:, :h],
                        name='decoded_adj_logits')(decoded)

    decoded_adj = Activation(activation='sigmoid', name='decoded_adj')(decoded_adj_logits)

    autoencoder = Model(
            inputs=[data], outputs=[decoded_adj, decoded_feats]
    )

    # compile the autoencoder
    adam = optimizers.Adam(lr=0.001, decay=0.0)

    autoencoder.compile(
        optimizer=adam,
        loss={'decoded_adj': masked_mean_squared_error,
              'decoded_feats': create_weighted_cosine_similarity_loss(weights=node_features_weight)},
        loss_weights={'decoded_adj': 3.0, 'decoded_feats': 1.0}
    )

    return encoder, autoencoder
