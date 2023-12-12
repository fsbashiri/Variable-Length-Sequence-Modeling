"""
Project: Variable Length Sequence Modeling
Branch:
Author: Azi Bashiri
Imported from Semi-supervised learning project on Feb. 2022
Last Modified: Oct. 2023
Description: A few useful functions and a config setting to use in different parts of the project.

"""
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import packages
import os
import subprocess
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras import metrics


cwd = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.abspath(os.path.join(cwd, os.pardir))

config = {
    # os environment
    'gpu_index': 0,  # GPU index to use. Negative value to run on CPU. List or a single integer

    # load data
    'save_pkl': False,  # save train/test data into pkl obj and copy sc_obj into data directory

    # data
    'n_features': 57,
    'sampling': '',  # options: None or '', 'down', 'up'. re-sampling the training encounters
    'drop_columns': ['outcome'],  # non-pred columns other than study_id; e.g ['outcome', 'time']
    'pred_type': 'seq2seq',  # ['seq2seq', 'seq2one']. seq2seq will pad labels and set the last return_sequences to True
    'validation_p': 0.2,  # validation split percentage if args.val_data == ''

    # Model
    'padding': 'post',  # ['pre', 'post']
    'mask_value': -1.0,  # padding value
    'add_weight': False,  # use sample_weight to balance outcome in training
    'transformation': 'norm',  # options: 'norm', 'std', 'ple_dt', None
    'n_bins': 10,  # Decision tree number of bins for ple_dt
    'tree_kwargs': {},  # Decision tree arguments for ple_dt
    'focal_bce_loss': False,  # Focal binary cross-entropy or not

    # HP tuner and training
    'batch_size': 64,
    'max_epoch': 20,  # Epoch to run
    'max_trials': 20,
    'num_initial_points': 10,  # for Bayesian Optimization
    'tuner_seed': 1474,
    'tuner_name': 'BayesianOptimization',  # options: RandomSearch, BayesianOptimization, Hyperband

    # logger: log_dir = log_path/log_folder. For default values set them to None or an empty string.
    # To resume tuning, direct tuner to the log_dir of an unfinished optimization by changing log_path and log_folder
    'log_path': "",  # str. Default: proj_dir/Output/
    'log_folder': ""  # str. Default format: log_YYYYmmdd-HHMM
}


def get_metrics(pred_type='cls'):
    """
    metrics to monitor during training and validation based on the type of predictions
    :param pred_type: 'cls' for classification, or None
    :return: list of preferred monitoring metrics
    """
    if pred_type.lower() == 'cls':
        ls_metrics = [metrics.BinaryAccuracy(name='accuracy'),
                      metrics.AUC(curve='ROC', name='auroc'),
                      metrics.AUC(curve='PR', name='auprc')]
    else:
        ls_metrics = None
    return ls_metrics


def auc_ci_delong(path_log_score="log_scores.csv"):
    """
    ROC CI (DeLong method). It uses subprocess to run an rscript
    :param path_log_score: address to the location of an log_scores.csv file, in which y_true and y_pred are stored.
    :return: out script from the subprocess
    """
    out = subprocess.check_output(
        ["Rscript", os.path.join(proj_dir, "Code", "VL052_roc_auc_ci_p_value.R"), "--i", path_log_score],
        universal_newlines=True)
    return out

