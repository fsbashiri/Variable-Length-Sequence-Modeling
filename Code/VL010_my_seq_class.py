"""
Project: Variable Length Sequence Modeling
Branch:
Author: Azi Bashiri
Last Modified: April 2023
Description: Python class VarLenSequence. It inherits properties of tensorflow.keras.utils.Sequence.
    VarLenSequence class can be used for: 1) padding instances of a batch in order to have a same sequence length,
    and 2) multi-processing and loading a large dataset into memory in batches rather than at once (which is not our
    focus in this project).
References:
    1. https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    2. Anoop's code for eCART4_RNNV4
    3. https://towardsdatascience.com/neural-network-for-input-of-variable-length-using-tensorflow-timedistributed-wrapper-a45972f4da51
"""
import numpy as np
from keras.utils import Sequence, pad_sequences


class VarLenSequence(Sequence):
    """Generates data for Keras
    A child class from Sequence requires two methods: __len__ and __getitem__
    The method __getitem__ should return a complete batch
    """
    def __init__(self, x_in, y_in, list_IDs, class_weights=None, batch_size=32, shuffle=True,
                 padding='post', pad_value=-10.0, expand_dim=False, pad_y=False):
        """Initialization
        :param x_in: numpy array of lists with variable length for input x
        :param y_in: numpy array of lists with variable length for input y (labels)
        :param list_IDs: list of all 'label' ids to use in the generator
        :param class_weights: list of all 'label' sample weights followed by a sample_weight 0 for padded timesteps
        :param batch_size: batch size at each iteration
        :param shuffle: True to shuffle label indexes after every epoch
        :param padding: type of padding. Options: 'pre' or 'post'
        :param pad_value: padding value used to pad shorter sequences
        :param expand_dim: Boolean, whether expand x input by axis 3 and 4. Used for TD-CNN modeling
        :param pad_y: Boolean, whether pad y. True for seq-to-seq predictions; False for seq-to-one predictions
        """
        self.x_in = x_in
        self.y_in = y_in
        self.list_IDs = list_IDs
        self.class_weights = class_weights
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.padding = padding
        self.pad_value = pad_value
        self.expand_dim = expand_dim
        self.pad_y = pad_y
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        """Denote the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch
        on_epoch_end is triggered once at the very beginning as well as at the end of each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :returns X and y
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        # print(f"  X: {X.shape}")  # for debugging only

        # compute sample_weights
        if isinstance(self.class_weights, list) or isinstance(self.class_weights, tuple):
            if self.pad_value == -1.0:
                # it works only if self.pad_value is -1.0; Meaning the last element in the class_weight list corresponds
                # to weights for padded time steps
                sample_weights = np.take(np.array(self.class_weights), np.round(y).astype('int'))
            else:
                sample_weights = np.zeros(y.shape)
                sample_weights[np.array(y) == self.pad_value] = self.class_weights[2]
                sample_weights[np.array(y) == 0] = self.class_weights[0]
                sample_weights[np.array(y) == 1] = self.class_weights[1]
            out = (X, y, sample_weights)
        else:
            # return only X and y if class_weights is None (i.e., you'd rather not apply weights to model training)
            out = (X, y)

        return out

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples
        :param list_IDs_temp: list of ids to load
        :returns X and y
        """
        # pad sequences of X. The output is numpy array
        X = pad_sequences(self.x_in[list_IDs_temp], padding=self.padding, value=self.pad_value, dtype='float32')
        if self.expand_dim:
            X = np.expand_dims(X, axis=3)
            X = np.expand_dims(X, axis=4)
        if self.pad_y:
            # always pad with zero to keep labels binary if using string 'accuracy' as metric;
            # pad with pad_value if using BinaryAccuracy() function as metric
            y = pad_sequences(self.y_in[list_IDs_temp], padding=self.padding, value=self.pad_value, dtype='int32')
        else:
            y = self.y_in[list_IDs_temp]
        return X, y
