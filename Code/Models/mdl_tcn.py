"""
Model: TCN
A TCN model for processing sequential data with varying lengths. It can handle both sequence-to-one classification
and sequence-to-sequence prediction. The 'pred_type' config determines the shape of the output predictions.
Please note:
    1. A Masking layer is added on top of the model
    2. input_shape=[None, n_features]
    3. return_sequences=True for all layers if pred_type is seq2seq; False for the last layer if otherwise
    4. output layer is TimeDistributed(Dense()) layer with sigmoid activation function for seq2seq prediction
    5. A subclass of TCN is defined to let TCN supports masking (i.e., passing mask to next layers).
    6. A tensorflow decorator is used to allow a keras model containing this subclass layer be saved/loaded properly

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
from keras_tuner.engine.hyperparameters import HyperParameters
from tcn import TCN
# current project directory
cwd = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(proj_dir)
import Code.VL099_utils as utils
import Code.VL011_globals as glb


# A subclass of TCN that passes the mask to the next layer
# reference: tds - How to write a custom keras model so that it can be deployed for serving
class MyTCN(layers.Layer):
    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 use_skip_connections=False,
                 dropout_rate=0.0,
                 return_sequences=False,
                 **kwargs):
        super(MyTCN, self).__init__(**kwargs)
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.tcn_layer = TCN(nb_filters=self.nb_filters,
                             kernel_size=self.kernel_size,
                             nb_stacks=self.nb_stacks,
                             dilations=self.dilations,
                             padding=self.padding,
                             use_skip_connections=self.use_skip_connections,
                             dropout_rate=self.dropout_rate,
                             return_sequences=self.return_sequences)
        # self.supports_masking = True  # pass mask to next layer

    def get_config(self):
        config = super(MyTCN, self).get_config()
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['dilations'] = self.dilations
        config['padding'] = self.padding
        config['use_skip_connections'] = self.use_skip_connections
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        return config

    def call(self, inputs, mask=None, training=None):
        output = self.tcn_layer(inputs)
        # output = super(MyTCN, self).call(inputs, training=training)
        # multiply mask with output to zero out output at padded time-steps
        if mask is not None and self.return_sequences:
            mask = tf.tile(tf.expand_dims(tf.cast(mask, 'float32'), -1), [1, 1, output.shape[-1]])
            output = output * mask
        return output

    def compute_mask(self, inputs, mask=None):
        if (mask is None) or (self.return_sequences is False):
            return None
        else:
            return mask


def create_model(hp):
    """
    Create n-layer TCN w/ dropout
    :param hp: keras-tuner hyperparameter object
    :return: a compiled keras model
    """
    # sequential model
    model = models.Sequential()
    model.add(layers.Masking(mask_value=utils.config['mask_value'], input_shape=(None, utils.config['n_features'])))
    # n_layers of {TCN -> Dropout}
    for i in range(hp.Int('num_layers', min_value=1, max_value=2, default=2)):  # set default to max_value
        nb_filters = hp.Int('nb_filters_' + str(i), min_value=10, max_value=50, default=24)
        kernel_size = hp.Int('kernel_size_' + str(i), min_value=2, max_value=10, default=3)
        nb_stacks = hp.Int('nb_stacks_' + str(i), min_value=1, max_value=6, default=2)
        dilations = hp.Int('dilations_len_' + str(i), min_value=2, max_value=9, default=4)  # [1, 2, 4, 8, 16, 32]
        use_skip_connections = hp.Boolean('use_skip_connections_' + str(i), default=False)
        dropout_rate = hp.Float('dropout_rate_' + str(i), min_value=0.0, max_value=0.8, default=0.05)
        model.add(MyTCN(nb_filters=nb_filters,
                        kernel_size=kernel_size,
                        nb_stacks=nb_stacks,
                        dilations=list(2 ** i for i in range(dilations)),
                        padding='causal',
                        use_skip_connections=use_skip_connections,
                        dropout_rate=dropout_rate,
                        return_sequences=True if (utils.config['pred_type'] == 'seq2seq' or (
                                (hp.get('num_layers') > 1) and (i + 1 < hp.get('num_layers')))) else False,
                        ))
    # output layer
    if utils.config['pred_type'] == 'seq2seq':
        model.add(layers.TimeDistributed(layers.Dense(1, activation='sigmoid'), name='TD_Dense'))
    else:
        model.add(layers.Dense(1, activation='sigmoid', name='Dense'))
    model.summary(print_fn=lambda x: glb.logger.log_string(x))

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
        optimizer = tf.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.25, clipvalue=0.25)
    elif opt == 'RMSProp':
        optimizer = tf.optimizers.RMSprop(learning_rate=lr_schedule, momentum=momentum, clipnorm=0.25, clipvalue=0.25)
    else:
        raise ValueError("unexpected optimizer name: %r" % (hp.get('optimizer'),))

    # compile the model
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryFocalCrossentropy() if utils.config['focal_bce_loss']
                  else tf.keras.losses.BinaryCrossentropy(),
                  # run_eagerly=True,
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
    glb.logger.log_string(f"y1_true: \n{y1_true} \ny1_pred: \n{y1_pred}")
    # evaluate
    eval = my_model.evaluate(x1, y1_true, return_dict=True)
    glb.logger.log_string(f"Evaluation metrics: {eval}")
    # close the log file
    glb.logger.log_fclose()
