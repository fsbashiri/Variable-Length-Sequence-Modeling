"""
Author: Azi Bashiri
Created: Oct. 2022
Last Modified: Oct. 2022
Description: A python script to examine causality of a model.
                It creates two test samples: one with n_ts time steps, and another one that is a copy of the first
                sample, but missing the information from the last n_mt time step. The predictions for both samples upto
                (n_ts - n_mt) time step should be the same; meaning that future measurements should not affect
                predictions from earlier time points.
                Causality can only be tested when pred_type is set to seq2seq.
Usage:  1. Review definitions right below import packages and modify them as needed
        2. In a terminal change directory to the project folder.
            Then type `python Code/Checkup_routines/examine_causality.py` and press Enter.

"""
# import packages
import os
import sys
import numpy as np
import pandas as pd
from keras_tuner.engine.hyperparameters import HyperParameters
cwd = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.join(cwd, os.pardir, os.pardir)
sys.path.append(proj_dir)
import Code.VL011_globals as glb
import Code.VL099_utils as utils

# definitions
n_ts = 14  # test sample sequence length
n_mt = 3  # number of missing time points
model_name = 'lstm_gru'  # ['tcn', 'lstm_gru', 'tdcnn']
# import create_model
if model_name == 'tcn':
    from Code.Models.mdl_tcn import create_model
elif model_name == 'lstm_gru':
    from Code.Models.mdl_lstm_gru import create_model
elif model_name == 'tdcnn':
    from Code.Models.mdl_tdcnn import create_model
else:
    raise AssertionError(f"Invalid model_name {model_name}.")


# GPU index -1: run on CPU and free up GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{-1}"

# Do not squeeze when printing
np.set_printoptions(threshold=sys.maxsize)  # print numpy arrays completely
pd.set_option("display.max_rows", None, "display.max_columns", None)  # print pandas dataframes completely


if __name__ == '__main__':
    # examine pred_type config
    if utils.config['pred_type'] == 'seq2one':
        raise AssertionError(f"Invalid value. 'pred_type' config is set to '{utils.config['pred_type']}'. "
                             f"Change it to 'seq2seq'. Causality can be tested for seq2seq predictions.")

    # start logging
    glb.init_logging(b_log_txt=True, log_name="log_causality_checkup.txt")
    glb.logger.log_string(f"Python version {sys.version.split(sep=' ')[0]} on {sys.platform} platform")
    glb.logger.log_string(glb.logger.__str__())

    # copy this file to log_dir
    os.system(f"cp {os.path.join(cwd, 'examine_causality.py')} {glb.logger.log_dir}")

    # create x1, x2
    x1 = np.random.rand(1, n_ts, utils.config['n_features'])
    x2 = np.delete(x1, np.s_[-n_mt:], axis=1)
    if model_name == 'tdcnn':
        # input shape for tdcnn is [None, None, n_features, 1, 1]
        x1 = np.expand_dims(np.expand_dims(x1, axis=-1), axis=-1)
        x2 = np.expand_dims(np.expand_dims(x2, axis=-1), axis=-1)

    # create model
    hps = HyperParameters()
    my_model = create_model(hps)

    # assess sequence predictions
    y1_pred = my_model.predict(x1)
    y2_pred = my_model.predict(x2)
    if np.all(y1_pred[:, :-n_mt, :] == y2_pred):
        glb.logger.log_string(f"The model is causal. Same sequence predictions upto the time step after which future "
                              f"measurements are missed.")
    else:
        glb.logger.log_string(f"!! WARNING: Model fails causality test. \n"
                              f"y1_pred = {y1_pred} \n"
                              f"y2_pred = {y2_pred} ")

    # close the log file
    glb.logger.log_fclose()
