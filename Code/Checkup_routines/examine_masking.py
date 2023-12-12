"""
Author: Azi Bashiri
Created: Oct. 2022
Last Modified: Mar. 2023
Description: A python script to examine padding and masking.
                It creates two test samples: one with n_ts time steps, and another one that is a copy of the first
                sample with one extra time step filled with mask_value. It can be pre-padded or post-padded. Both
                samples are taken in by a model, which outputs predictions. These predictions are compared and examined.
                Masking can only be tested when pred_type is set to seq2seq.
Usage:  1. Review definitions right below import packages and modify them as needed
        2. In a terminal change directory to the project folder.
            Then type `python Code/Checkup_routines/examine_masking.py` and press Enter.

"""
# import packages
import os
import sys
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # GitHub tensorflow issue #59779
cwd = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(proj_dir)
import Code.VL011_globals as glb
import Code.VL099_utils as utils
# run on CPU and free up GPU - call before importing tensorflow
if utils.config['gpu_index'] < 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{utils.config['gpu_index']}"
from keras_tuner.engine.hyperparameters import HyperParameters

# definitions
n_ts = 10  # test sample sequence length
model_name = 'lstm_gru'  # ['tcn', 'lstm_gru', 'tdcnn']
# import create_model
if model_name == 'tcn':
    from Code.Models.mdl_tcn import create_model, MyTCN
elif model_name == 'lstm_gru':
    from Code.Models.mdl_lstm_gru import create_model
elif model_name == 'tdcnn':
    from Code.Models.mdl_tdcnn import create_model
else:
    raise AssertionError(f"Invalid model_name {model_name}.")


# Do not squeeze when printing
np.set_printoptions(threshold=sys.maxsize)  # print numpy arrays completely
pd.set_option("display.max_rows", None, "display.max_columns", None)  # print pandas dataframes completely


if __name__ == '__main__':
    # examine pred_type config
    if utils.config['pred_type'] == 'seq2one':
        raise AssertionError(f"Invalid value. 'pred_type' config is set to '{utils.config['pred_type']}'. "
                             f"Change it to 'seq2seq'. Causality can be tested for seq2seq predictions.")

    # start logging
    glb.init_logging(b_log_txt=True, log_name="log_masking_checkup.txt")
    glb.logger.log_string(f"Python version {sys.version.split(sep=' ')[0]} on {sys.platform} platform")
    glb.logger.log_string(glb.logger.__str__())
    glb.logger.log_string(f": config: \n{utils.config}")

    # copy this file to log_dir
    os.system(f"cp {os.path.join(cwd, 'examine_masking.py')} {glb.logger.log_dir}")
    os.system(f"cp {os.path.join(proj_dir, 'Code', 'Models', 'mdl_' + model_name + '.py')} {glb.logger.log_dir}")

    # create x1, x2, y1_true, y2_true
    x1 = np.random.rand(1, n_ts, utils.config['n_features'])
    y1_true = np.reshape(np.random.choice([0, 1], size=n_ts, replace=True), (1, n_ts, 1))
    x2 = x1.copy()
    if utils.config['padding'] == 'pre':
        x2 = np.append(utils.config['mask_value'] * np.ones((1, 1, utils.config['n_features'])), x2, axis=1)
        y2_true = np.append(utils.config['mask_value'] * np.ones((1, 1, 1)), y1_true, axis=1).astype(int)
    else:
        x2 = np.append(x2, utils.config['mask_value'] * np.ones((1, 1, utils.config['n_features'])), axis=1)
        y2_true = np.append(y1_true, utils.config['mask_value'] * np.ones((1, 1, 1)), axis=1).astype(int)
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
    if model_name == 'tdcnn':
        # CNN doesn't support masking. Mask is computed from input and passed to LSTM through the mask input argument
        mask = my_model(x2)._keras_mask.numpy()  # functional API
    else:
        # tcn issue #240: tcn output object has no attribute _keras_mask. For now, I've put a patch with subclassing tcn
        mask = my_model.layers[0].compute_mask(x2).numpy()

    # *** EXAMINATION 1: same predictions at time steps with mask == True
    glb.logger.log_string(f"\nExamination 1")
    if np.all(y1_pred == y2_pred[mask]):
        glb.logger.log_string(f"Same predictions at time steps that were not padded. \n"
                              f"y1_pred == y2_pred = \n{y1_pred}")
    else:
        glb.logger.log_string(f"!! WARNING: sequence predictions don't match.\n"
                              f"flattened y1_pred (shape: {y1_pred.shape}) = \n{y1_pred.ravel()}\n"
                              f"flattened y2_pred (shape: {y2_pred.shape}) = \n{y2_pred.ravel()}")

    # *** EXAMINATION 2: predictions for time steps when mask == False
    indx = mask.argmin()
    glb.logger.log_string(f"\nExamination 2")
    if utils.config['padding'] == 'pre':
        if y2_pred[np.equal(mask, False)] == 0.5:
            glb.logger.log_string(f"Pre-padded sequences are masked. Predictions for those sequences are equal to 0.5.")
        else:
            glb.logger.log_string(f"!! WARNING: Pre-padded sequences are not masked. Expected predictions 0.5.")
    else:
        if np.all(y2_pred[0, indx-1, 0] == y2_pred[0, indx:, 0]):
            glb.logger.log_string(f"Propagated sequence prediction for time steps that were masked.")
        else:
            glb.logger.log_string(f"!! WARNING: prediction propagation for padded time steps didn't work.")

    # *** EXAMINATION 3: assess evaluation metrics except loss!
    eval1 = my_model.evaluate(x1, y1_true, return_dict=True)
    eval2 = my_model.evaluate(x2, y2_true, return_dict=True)
    glb.logger.log_string(f"\nExamination 3")
    for key in my_model.metrics_names[1:]:
        if eval1[key] == eval2[key]:
            glb.logger.log_string(f"{key}: y1 = y2 = {eval1[key]:.4f}")
        else:
            glb.logger.log_string(f"!! WARNING: evaluation metric {key} do not match.\n"
                                  f"eval1['{key}']={eval1[key]} - eval2['{key}']={eval2[key]}")
    # loss will not be the same. Just print it out
    glb.logger.log_string(f"To note: \nloss: x1->{eval1['loss']}  x2->{eval2['loss']}")

    # ********************** train model and repeat EXAMINATIONs 1-3
    glb.logger.log_string(f"\nTraining model with x2")
    my_model.fit(x2, y2_true, epochs=5, batch_size=1)
    glb.logger.log_string(f": Training history:\n{pd.DataFrame(my_model.history.history)}")
    y1_pred_trained = my_model.predict(x1)
    y2_pred_trained = my_model.predict(x2)

    # *** EXAMINATION 1: same predictions when mask == True
    glb.logger.log_string(f"\nRepeat Examination 1")
    if np.all(y1_pred_trained == y2_pred_trained[mask]):
        glb.logger.log_string(f"Same predictions at time steps that were not padded. \n"
                              f"y1_pred_trained == y2_pred_trained[not_padded] = \n{y1_pred_trained}")
    else:
        glb.logger.log_string(f"!! WARNING: sequence predictions don't match.\n"
                              f"flattened new y1_pred (shape: {y1_pred_trained.shape}) = \n{y1_pred_trained.ravel()}\n"
                              f"flattened new y2_pred (shape: {y2_pred_trained.shape}) = \n{y2_pred_trained.ravel()}")

    # *** EXAMINATION 2: predictions for time steps when mask == False
    glb.logger.log_string(f"\nRepeat Examination 2")
    if utils.config['padding'] == 'pre':
        if y2_pred_trained[np.equal(mask, False)] == 0.5:
            glb.logger.log_string(f"Pre-padded sequences are masked. Predictions for those sequences are equal to 0.5.")
        else:
            glb.logger.log_string(f"!! WARNING: Pre-padded sequences are not equal to 0.5.")
    else:
        if np.all(y2_pred_trained[0, indx - 1, 0] == y2_pred_trained[0, indx:, 0]):
            glb.logger.log_string(f"Propagated sequence prediction for time steps that were masked.")
        else:
            glb.logger.log_string(f"!! WARNING: prediction propagation for padded time steps don't work.")

    # *** EXAMINATION 3: assess evaluation metrics except loss!
    eval1 = my_model.evaluate(x1, y1_true, return_dict=True)
    eval2 = my_model.evaluate(x2, y2_true, return_dict=True)
    glb.logger.log_string(f"\nRepeat Examination 3")
    for key in my_model.metrics_names[1:]:
        if eval1[key] == eval2[key]:
            glb.logger.log_string(f"{key}: y1_pred_trained = y2_pred_trained = {eval1[key]:.4f}")
        else:
            glb.logger.log_string(f"!! WARNING: evaluation metric {key} do not match.\n"
                                  f"eval1['{key}']={eval1[key]} - eval2['{key}']={eval2[key]}")
    # loss will not be the same. Just print it out
    glb.logger.log_string(f"To note: \nloss: x1->{eval1['loss']}  x2->{eval2['loss']}")

    # close the log file
    glb.logger.log_fclose()
