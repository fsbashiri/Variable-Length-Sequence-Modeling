"""
Model: LSTM/GRU

An LSTM/GRU model for processing sequential data with varying lengths. It can handle both sequence-to-one classification
and sequence-to-sequence prediction. The 'pred_type' config determines the shape of the output predictions.
Please note:
    1. A Masking layer is added on top of the model
    2. input_shape=[None, n_features]
    3. return_sequences=True for all layers before the last layer. It's True for seq2seq prediction and False otherwise
    4. output layer is TimeDistributed(Dense()) layer with sigmoid activation function for seq2seq prediction

Author: Azi Bashiri
Date: July 2022
Last Modified: Oct. 2023
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras import models, layers, metrics
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from keras.regularizers import L1L2
from keras_tuner.engine.hyperparameters import HyperParameters
# current project directory
cwd = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(proj_dir)
import Code.VL099_utils as utils
import Code.VL011_globals as glb


def create_model(hp):
    """
    Create n-layer LSTM/GRU w/ dropout
    :param hp: keras-tuner hyperparameter object
    :return: a compiled keras model
    """
    # instance of Mirrored Strategy
    mirrored_strategy = tf.distribute.MirroredStrategy()

    # Open strategy scope
    with mirrored_strategy.scope():
        # create n-layer LSTM or GRU w/ dropout based on hp
        model = models.Sequential()
        model.add(layers.Masking(mask_value=utils.config['mask_value'], input_shape=(None, utils.config['n_features'])))
        for i in range(hp.Int('num_layers', min_value=1, max_value=2, default=2)):  # set default to max_value
            cell_type = hp.Choice('cell_type_' + str(i), ['LSTM', 'GRU'], default='LSTM')
            units = hp.Int('units_' + str(i), min_value=20, max_value=300, default=20)  # smaller max_value prevent OOM
            dropout = hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.9, default=0.1)
            # requirement of the cuDNN kernel
            rec_dropout = 0.0
            activation = 'tanh'
            if cell_type == 'LSTM':
                model.add(layers.LSTM(units,
                                      dropout=dropout,
                                      recurrent_dropout=rec_dropout,
                                      activation=activation,
                                      # kernel_regularizer=L1L2(l1=0.0001, l2=0.001),
                                      return_sequences=True if (utils.config['pred_type'] == 'seq2seq' or (
                                              (hp.get('num_layers') > 1) and (i + 1 < hp.get('num_layers')))) else False))
            elif cell_type == 'GRU':
                model.add(layers.GRU(units,
                                     dropout=dropout,
                                     recurrent_dropout=rec_dropout,
                                     activation=activation,
                                     # kernel_regularizer=L1L2(l1=0.0001, l2=0.001),
                                     return_sequences=True if (utils.config['pred_type'] == 'seq2seq' or (
                                             (hp.get('num_layers') > 1) and (i + 1 < hp.get('num_layers')))) else False))
            else:
                raise ValueError("unexpected cell type: %r" % cell_type)
            model.add(layers.TimeDistributed(layers.LayerNormalization(), name='TD_LN_' + str(i)))
        # output layer
        if utils.config['pred_type'] == 'seq2seq':
            model.add(layers.TimeDistributed(layers.Dense(1, activation='sigmoid'), name='TD_Dense'))
        else:
            model.add(layers.Dense(1, activation='sigmoid', name='Dense'))
        model.summary(print_fn=lambda x: glb.logger.log_string(x))

        # hyperparameters related to the optimizer
        opt = hp.Choice('optimizer', ['Adam', 'RMSProp'], default='Adam')  # no 'SGD',
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
    # log a string
    glb.init_logging(b_log_txt=True, log_name="my_log.txt")
    # create x1 and y1_true
    x1 = np.random.rand(1, 10, utils.config['n_features'])
    if utils.config['pred_type'] == 'seq2seq':
        y1_true = np.reshape(np.random.choice([0, 1], size=10, replace=True), (1, 10, 1))
    else:
        y1_true = np.random.choice([0, 1], size=1)
    # create a model
    hps = HyperParameters()
    my_model = create_model(hps)
    # predict
    y1_pred = my_model.predict(x1)
    eval = my_model.evaluate(x1, y1_true, return_dict=True)
    glb.logger.log_string(f"y1_true: \n{y1_true} \ny1_pred: \n{y1_pred}")
    glb.logger.log_string(f"Evaluation metrics: {eval}")
    # close the log file
    glb.logger.log_fclose()

