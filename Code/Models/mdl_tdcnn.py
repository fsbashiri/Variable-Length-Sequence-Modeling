"""
Model: Time Distributed CNN
A TD-CNN model for processing sequential data with varying lengths. It can handle both sequence-to-one classification
and sequence-to-sequence prediction. The 'pred_type' config determines the shape of the output predictions.
Please note:
    1. It uses functional API model. It cannot be a sequential model
    1. A Masking layer is added to pass the mask from compute_mask function to the LSTM layers
    2. input_shape=[None, n_features, 1, 1]
    3. return_sequences=True for all layers if pred_type is seq2seq; False for the last layer if otherwise
    4. output layer is TimeDistributed(Dense()) layer with sigmoid activation function for seq2seq prediction
    5. The BatchNormalization layer that comes after Conv2D layer doesn't precede a TimeDistributed layer. If it does,
        it will throw a CUDNN_STATUS_NOT_SUPPORTED error.

Author: Azi Bashiri
Date: Jan 2023
Last Modified: Oct. 2023
"""

import os
import sys
import numpy as np
from datetime import datetime
import tensorflow as tf
import keras.backend as k
from keras import models, layers, metrics
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from keras_tuner.engine.hyperparameters import HyperParameters
# current project directory
cwd = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(proj_dir)
import Code.VL011_globals as glb
import Code.VL099_utils as utils


def create_model(hp):
    """
    Create n-layer TD-CNN followed by n-layer LSTM/GRU w/ dropout
    :param hp: keras-tuner hyperparameter object
    :return: a compiled keras model
    """
    # instance of Mirrored Strategy
    mirrored_strategy = tf.distribute.MirroredStrategy()

    # open strategy scope
    with mirrored_strategy.scope():
        # Functional API model
        input = layers.Input(shape=[None, utils.config['n_features'], 1, 1])
        masking_input = tf.squeeze(input, [3, 4])  # squeeze axis 3,4 for masking layer
        mask = layers.Masking(mask_value=utils.config['mask_value']).compute_mask(masking_input)
        m = tf.identity(input, name="Identity")  # copy of input
        # create n-layer TD-CNN based on hp
        for i in range(hp.Int('num_tdcnn_layers', min_value=1, max_value=2, default=2)):
            filters = hp.Int('filters_' + str(i), min_value=5, max_value=30, default=16)
            kernel_size = hp.Int('kernel_size_' + str(i), min_value=2, max_value=4, default=3)
            strides = hp.Int('strides_' + str(i), min_value=1, max_value=2, default=1)
            activation = hp.Choice('activation_cnn_' + str(i), ['tanh', 'sigmoid', 'relu'], default='relu')
            m = layers.TimeDistributed(layers.Conv2D(filters=filters,
                                                     kernel_size=(kernel_size, 1),
                                                     strides=(strides, 1),
                                                     activation=activation), name='TD_Conv2D_' + str(i))(m)
            m = layers.BatchNormalization(name='CONV_BN_' + str(i))(m)
        m = layers.TimeDistributed(layers.AvgPool2D(pool_size=(2, 1), strides=None), name='TD_AvgPool')(m)
        m = layers.TimeDistributed(layers.Flatten(), name='TD_Flatten')(m)
        # create n-layer LSTM or GRU w/ dropout based on hp
        for i in range(hp.Int('num_rnn_layers', min_value=1, max_value=2, default=2)):
            cell_type = hp.Choice('cell_type_' + str(i), ['LSTM', 'GRU'], default='LSTM')
            units = hp.Int('units_' + str(i), min_value=20, max_value=250, default=20)
            dropout = hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.8, default=0.1)
            # requirement of the cuDNN kernel
            rec_dropout = 0.0
            activation = 'tanh'
            if cell_type == 'LSTM':
                m = layers.LSTM(units,
                                dropout=dropout,
                                recurrent_dropout=rec_dropout,
                                activation=activation,
                                return_sequences=True if (utils.config['pred_type'] == 'seq2seq' or (
                                              (hp.get('num_rnn_layers') > 1) and (i + 1 < hp.get('num_rnn_layers'))))
                                else False,
                                name='LSTM_' + str(i)
                                )(m, mask=mask)
            elif cell_type == 'GRU':
                m = layers.GRU(units,
                               dropout=dropout,
                               recurrent_dropout=rec_dropout,
                               activation=activation,
                               return_sequences=True if (utils.config['pred_type'] == 'seq2seq' or (
                                              (hp.get('num_rnn_layers') > 1) and (i + 1 < hp.get('num_rnn_layers'))))
                               else False,
                               name='GRU_' + str(i)
                               )(m, mask=mask)
            else:
                raise ValueError("unexpected cell type: %r" % cell_type)
            m = layers.TimeDistributed(layers.LayerNormalization(), name='TD_RNN_LN_' + str(i))(m)
        # output layer
        if utils.config['pred_type'] == 'seq2seq':
            output = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'), name='TD_Dense')(m)
        else:
            output = layers.Dense(1, activation='sigmoid', name='Dense')(m)
        model = models.Model(inputs=input, outputs=output)
        model.summary(print_fn=lambda x: glb.logger.log_string(x), line_length=120)

        # hyper-params related to the optimizer
        opt = hp.Choice('optimizer', ['Adam', 'RMSProp'], default='RMSProp')
        momentum = hp.Float('momentum', min_value=0.0, max_value=1.0, default=0.0)
        # define a learning_rate schedule: ExponentialDecay
        lr_base = hp.Float('lr', min_value=0.0001, max_value=0.001, default=0.001)
        lr_decay_scheduling = hp.Boolean('lr_decay_scheduling', default=True)
        lr_decay_nstep = hp.Int('lr_decay_nstep', min_value=300, max_value=500, default=300)
        lr_decay_rate = hp.Float('lr_decay_rate', min_value=0.7, max_value=0.99, default=0.95)
        # optimizers accept both a fixed lr value and a lr_schedule as input
        if lr_decay_scheduling is True:
            lr_schedule = ExponentialDecay(lr_base,
                                           decay_steps=lr_decay_nstep,
                                           decay_rate=lr_decay_rate,
                                           staircase=True)
        else:
            lr_schedule = lr_base

        # setting up the optimizer
        if opt == 'Adam':
            optimizer = tf.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.5, clipvalue=0.5)
        elif opt == 'RMSProp':
            optimizer = tf.optimizers.RMSprop(learning_rate=lr_schedule, momentum=momentum,
                                              clipnorm=0.5, clipvalue=0.5)
        else:
            raise ValueError("unexpected optimizer name: %r" % (hp.get('optimizer'),))

        # compile the model
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.BinaryFocalCrossentropy() if utils.config['focal_bce_loss']
                      else tf.keras.losses.BinaryCrossentropy(),
                      sample_weight_mode='temporal' if utils.config['add_weight'] else None,
                      metrics=[metrics.BinaryAccuracy(name='accuracy'),
                               metrics.AUC(curve='ROC', name='auroc', num_thresholds=1000),
                               metrics.AUC(curve='PR', name='auprc', num_thresholds=1000)])

    # log HP config
    glb.logger.log_string(f"{datetime.today().strftime('%Y.%m.%d %H:%M:%S')} HP config: {hp.get_config()['values']}\n")
    return model


def log_string_search_space_summary(tuner):
    """
    The built-in search_space_summary method does not return a printable string. I re-wrote it with log_string
    :param tuner: keras-tuner tuner object
    :return: None
    """
    hp = tuner.oracle.get_space()
    glb.logger.log_string(f"Default search space size: {len(hp.space)}")
    for p in hp.space:
        config = p.get_config()
        name = config.pop('name')
        glb.logger.log_string(f"- {name} ({p.__class__.__name__})")
        glb.logger.log_string(f"\t{config}")
    return None


if __name__ == '__main__':
    # run on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{-1}"
    # start logging
    glb.init_logging(b_log_txt=True, log_name="my_log.txt")
    # create x1 and y1_true
    x1 = np.random.rand(1, 10, utils.config['n_features'], 1, 1)
    if utils.config['pred_type'] == 'seq2seq':
        y1_true = np.reshape(np.random.choice([0, 1], size=10, replace=True), (1, 10, 1))
    else:
        y1_true = np.random.choice([0, 1], size=1)
    # create a model
    hps = HyperParameters()
    my_model = create_model(hps)
    # predict
    y1_pred = my_model.predict(x1)
    glb.logger.log_string(f"y1_true: \n{y1_true} \ny1_pred: \n{y1_pred}")
    # evaluate
    eval = my_model.evaluate(x1, y1_true, return_dict=True)
    glb.logger.log_string(f"Evaluation metrics: {eval}")
    # close the log file
    glb.logger.log_fclose()
