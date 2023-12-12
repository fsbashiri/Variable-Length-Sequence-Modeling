"""
Project: Variable Length Sequence Modeling
Branch:
Author: Azi Bashiri
Imported from Semi-supervised learning project on Feb. 2022
Last Modified: Oct. 2023
Description: Python script for training a DL-base model using a given dataset. The training section assumes data has
                variable length sequences of measurements.
Notes:  1. Review 'config' setup in the VL099_utils.py script and modify its values as needed
        2. Use the command line arguments to change: model type and path to training file, validation file, test file
        3. In a terminal change directory to the project folder.
            Then type `python Code/VL020_train.py [-h]` and press Enter.
"""

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import packages
import datetime
import os
import sys
import timeit
import pickle
import argparse
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # GitHub tensorflow issue #59779
cwd = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.abspath(os.path.join(cwd, os.pardir))
sys.path.append(proj_dir)
sys.path.append(os.path.abspath(os.path.join(proj_dir, os.pardir, 'rtdl')))
from Code.VL010_my_seq_class import VarLenSequence
import Code.VL099_utils as utils
import Code.VL011_globals as glb
from Code.Data_loader.data_loader import DataLoader
from rtdl.data import PiecewiseLinearEncoder
# Processing unit: GPU or CPU. Do this before importing keras or tensorflow
if isinstance(utils.config['gpu_index'], int) and utils.config['gpu_index'] < 0:  # Negative value will run in CPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
# else:
#     os.environ["CUDA_VISIBLE_DEVICES"] = f"{utils.config['gpu_index']}"
import tensorflow as tf
if (isinstance(utils.config['gpu_index'], list) or
        (isinstance(utils.config['gpu_index'], int) and utils.config['gpu_index'] >= 0)):
    # Restrict TensorFlow to only use one GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if isinstance(utils.config['gpu_index'], list):
        devices = [physical_devices[gpu_id] for gpu_id in utils.config['gpu_index']]
    else:  # instance of int
        devices = physical_devices[utils.config['gpu_index']]
    tf.config.set_visible_devices(devices, 'GPU')
    print(f'devices: {tf.config.list_logical_devices("GPU")}')
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras_tuner import Objective
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.tuners import RandomSearch, BayesianOptimization, Hyperband
from keras_tuner.engine import tuner


# random seed
np.random.seed(777)
tf.random.set_seed(42)

# Do not squeeze when printing
np.set_printoptions(threshold=sys.maxsize)  # print numpy arrays completely
pd.set_option("display.max_rows", None, "display.max_columns", None)  # print pandas dataframes completely


def start_logging():
    # ***** Start logging *****
    glb.init_logging(b_log_txt=True, log_name="log_train.txt",
                     log_path=utils.config['log_path'],
                     log_folder=utils.config['log_folder'])

    # copy files for backup.
    os.system(f"cp {os.path.join(proj_dir, 'Code', 'VL020_train.py')} {glb.logger.log_dir}")  # bkp of train procedure
    os.system(f"cp {os.path.join(proj_dir, 'Code', 'Models', model_choice)} {glb.logger.log_dir}")  # bkp of model
    os.system(f"cp {os.path.join(proj_dir, 'Code', 'VL010_my_seq_class.py')} {glb.logger.log_dir}")  # bkp of seq class
    os.system(f"cp {os.path.join(proj_dir, 'Code', 'VL099_utils.py')} {glb.logger.log_dir}")  # bkp of utils
    glb.logger.log_string('Python %s on %s' % (sys.version, sys.platform))
    glb.logger.log_string(glb.logger.__str__())


def parse_args():
    parser = argparse.ArgumentParser(description="Training a model for processing sequential data with varying lengths",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model",
                        default="lstm_gru", type=str,
                        help="Model of choice. Default: 'lstm_gru' \n"
                             "Options: \n\t"
                             "'tcn'= temporal convolution network \n\t"
                             "'lstm_gru'= multi-layer LSTM and GRU \n\t"
                             "'tdcnn'= time distributed wrapper with CNN")
    parser.add_argument("-t", "--train_data",
                        default="", type=str,
                        help="Path to training data. Default: ''")
    parser.add_argument("-v", "--val_data", nargs='?', const='',
                        default="", type=str,
                        help="Path to validation data. Default: ''"
                             "\nLeave empty if validation data should be split from training data")
    parser.add_argument("-e", "--test_data",
                        default="", type=str,
                        help="Path to hold-out test data. Default: ''")
    parser.add_argument("-d", "--data_dir",
                        default="", type=str,
                        help="Path to data directory for fast loading pickle object. Default: ''")
    return parser.parse_args()


def prep_data_df_to_array(df):
    """
    Take in a dataframe containing predictor variables, outcome, study_id, and possibly other non-pred variables.
    Extract x and y, and reorder into numpy array of arrays
    :param df: input dataframe
    :return: x_out: array of shape (n_patients,) where x_out[i] is an array from the i-th encounter
    :return: y_out: array of shape () if pred_type == seq2one; and array of shape (n_patients, ) otherwise
    """
    # first extract y out of dataframe
    if utils.config['pred_type'] == 'seq2one':
        y_out = df[['study_id', 'outcome']].groupby('study_id').max()['outcome'].to_numpy()
    else:
        g = df.groupby('study_id').cumcount()
        y_out = df.set_index(['study_id', g])[['outcome']].groupby(level=0).apply(lambda x: x.values).\
            to_numpy(dtype=object)
    # drop non-pred columns except study_id
    df.drop(utils.config['drop_columns'], axis=1, inplace=True)
    # reorder x into numpy array of list
    g = df.groupby('study_id').cumcount()
    x_out = df.set_index(['study_id', g]).groupby(level=0).apply(lambda x: x.values).to_numpy(dtype=object)
    return x_out, y_out


# global variables related to CustomLoggingCallback
log_metrics = {}  # to collect the best metrics (wrt es1 objective) for each run
log_trialids = {}  # only needed when tuning with Hyperband
search_num = [0]


class CustomLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(CustomLoggingCallback, self).__init__()
        self.epoch_times = []
        self.epoch_time_start = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = timeit.default_timer()

    def on_epoch_end(self, epoch, logs=None):
        # get the most related Trial_id
        t_id = tuner.oracle.ongoing_trials[tuner.tuner_id].trial_id  # current trial id
        t_state = tuner.oracle.trials.get(t_id).get_state()  # current tuner state
        if 'tuner/trial_id' in t_state['hyperparameters']['values'].keys():
            # this happens only to Hyperband when the tuner picks up one of the previous trials
            old_t_id = t_state['hyperparameters']['values']['tuner/trial_id']
            # this happens when a trial_id is connected to a second or higher degree trial_id
            t_id_list = [log_trialids[key] for key in log_trialids]
            for i, sublist in enumerate(t_id_list):
                if old_t_id in sublist:
                    if t_id not in sublist:
                        # append new t_id if it's not already in the sublist
                        log_trialids[[*log_trialids][i]].append(t_id)
                    t_id = [*log_trialids][i]  # update t_id to the first degree t_id
                    break
        # log logs
        glb.logger.log_string(f"{datetime.datetime.today().strftime('%Y.%m.%d %H:%M:%S')} **** Epoch {epoch + 1:03}: " +
                              ' - '.join(f"{m}: {v:.4f}" for m, v in zip(logs.keys(), logs.values())))
        if epoch == 0:
            # add a new key to the dict when a new training starts
            log_metrics[t_id] = [epoch + 1, *logs.values()]
            log_trialids[t_id] = [t_id]
        else:
            # replace metrics if it's improved
            if log_metrics[t_id][-2] < logs['val_auroc']:
                log_metrics[t_id] = [epoch + 1, *logs.values()]
        self.epoch_times.append(round(timeit.default_timer() - self.epoch_time_start))  # in seconds
        glb.logger.log_string(f"Time for Epoch {epoch + 1}: {self.epoch_times[-1]} sec")
    # def on_batch_begin(self, batch, logs=None):  # for debugging purposes
    #     if batch % 1000 == 0:
    #         glb.logger.log_string(f"Begin batch {batch}")
    #     current_decayed_lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
    #     glb.logger.log_string(f"Batch: {batch} - lr: {current_decayed_lr}")

    def on_train_end(self, logs=None):
        search_num[0] += 1
        mean_time = round(np.mean(self.epoch_times) / 60, 2)  # in minutes
        glb.logger.log_string(f"Average time per epoch = {mean_time} minutes")
        glb.logger.log_string(f"Search {search_num[0]}/{utils.config['max_trials']} Completed.\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # input arguments
    args = parse_args()
    if np.isin(args.model, ['tcn', 'tdcnn', 'lstm_gru']):
        model_choice = "mdl_" + args.model + ".py"
    else:
        raise AssertionError(f"Unknown model of choice: {args.model}")
    if os.path.isfile(args.train_data) or (not args.train_data):  # args can be an existing file or an empty string
        train_file = args.train_data
    else:
        raise FileNotFoundError(f"train_data = {args.train_data}")
    if os.path.isfile(args.val_data) or (not args.val_data):  # args.val_data can be empty
        val_file = args.val_data
    else:
        raise FileNotFoundError(f"val_data = {args.val_data}")
    if os.path.isfile(args.test_data) or (not args.test_data):
        test_file = args.test_data
    else:
        raise FileNotFoundError(f"test_data = {args.test_data}")
    if os.path.isdir(args.data_dir) or (not args.data_dir):
        data_dir = args.data_dir
    else:
        raise NotADirectoryError(f"data_dir = {args.data_dir}")
    # source of data: read from csv (i.e., long_load) or load a pickle obj (i.e., fast_load)
    if (len(train_file) > 0) and (len(test_file) > 0):
        read_from_csv = True
    elif len(data_dir) > 0:
        read_from_csv = False
    else:
        raise AssertionError(f"Invalid input arguments. Two ways to direct the code to input data: 1) use -t and -e "
                             f"arguments to point to train_data and test_data [.csv] files; 2) use -d argument to "
                             f"point to a valid directory with a pickle object that contains the pre-processed train "
                             f"and test data.")

    # load model
    if model_choice == 'mdl_lstm_gru.py':
        from Code.Models.mdl_lstm_gru import create_model, log_string_search_space_summary, CausalSelfAttention
    elif model_choice == 'mdl_tcn.py':
        from Code.Models.mdl_tcn import create_model, log_string_search_space_summary, MyTCN, CausalSelfAttention2
    elif model_choice == 'mdl_tdcnn.py':
        from Code.Models.mdl_tdcnn import create_model, log_string_search_space_summary

    # ***** Start logging *****
    start_logging()
    glb.logger.log_string(f": config: \n{utils.config}")
    glb.logger.log_string(f": Arguments: \n{str(args)[10:-1]}")

    # ***** get the data
    glb.logger.log_string('\n---------- Preparing data')
    glb.logger.log_string('======================================\n')
    # sc_path
    if utils.config['transformation'] == 'norm':
        scaling = "min_max"
        sc_path = os.path.join(glb.logger.log_dir, 'scaling_obj_norm.pkl')
    elif utils.config['transformation'] == 'std':
        scaling = "mean_std"
        sc_path = os.path.join(glb.logger.log_dir, 'scaling_obj_std.pkl')
    elif utils.config['transformation'] == 'ple_dt':
        scaling = None  # ple_dt
        sc_path = os.path.join(glb.logger.log_dir, 'scaling_obj_ple_dt.pkl')
    elif utils.config['transformation'] is None:
        scaling = None
        sc_path = None
    else:  # unrecognized option
        raise ValueError(f"Unsupported transfromation {utils.config['transformation']}. "
                         f"Accepted options are: 'norm', 'std', 'ple_dt', None.")
    # read from csv OR load from pickle
    if read_from_csv:
        glb.logger.log_string(f": Training data")
        data_in = DataLoader(filenames=[train_file],
                             target="outcome",
                             loadcreate01=1,
                             sc_path=sc_path,
                             scaling=scaling)
        data_in.get_data_short(print_stats=False)
        df_input = data_in.data
        if utils.config['transformation'] == 'ple_dt':
            glb.logger.log_string(f"\n: creating a PLE_DT with {utils.config['n_bins']} bins")
            ple_dt = PiecewiseLinearEncoder('decision_tree',
                                            dict(n_bins=utils.config['n_bins'], regression=True,
                                                 tree_kwargs=utils.config['tree_kwargs']),
                                            stack=False)
            # fit ple_dt on training samples
            ple_dt.fit(df_input[[*data_in.pred_vars]].to_numpy(),
                       df_input[[data_in.target]].to_numpy())
            with open(sc_path, 'wb') as f:
                pickle.dump(ple_dt, f)
                glb.logger.log_string(f"\t Scaler object is saved at: {sc_path}")
            Xtrans = ple_dt.transform(df_input[[*data_in.pred_vars]].to_numpy())  # faster conversion to numpy array
            utils.config['n_features'] = Xtrans.shape[1]
            glb.logger.log_string(f": n_features updated to Xtrans.shape[1]={Xtrans.shape[1]}")
            X = pd.concat([df_input[['study_id', 'outcome']],
                           pd.DataFrame(Xtrans)], axis=1)
            df_input = X.copy()
        # convert df to array
        x_derivation, y_derivation = prep_data_df_to_array(df_input)

        # test data
        glb.logger.log_string(f"\n: Test data")
        data_in = DataLoader(filenames=[test_file],
                             target="outcome",
                             loadcreate01=0,
                             sc_path=sc_path,
                             scaling=scaling)
        data_in.get_data_short(print_stats=False)
        df_input = data_in.data
        if utils.config['transformation'] == 'ple_dt':
            # apply transformation on test samples
            glb.logger.log_string(f": transforming test set using PLE_DT")
            Xtrans = ple_dt.transform(df_input[[*data_in.pred_vars]].to_numpy())
            X = pd.concat([df_input[['study_id', 'outcome']],
                           pd.DataFrame(Xtrans)], axis=1)
            df_input = X.copy()
        x_test, y_test = prep_data_df_to_array(df_input)

        if utils.config['save_pkl']:
            # save data for fast load
            pkl_fname = "VLS_impute_train_test_" + utils.config['pred_type'] + "_" + \
                        utils.config['transformation'] + ".pkl"
            with open(os.path.join(os.path.dirname(train_file), pkl_fname), "wb") as f:
                pickle.dump([x_derivation, y_derivation, x_test, y_test], f)
            glb.logger.log_string(f"\n: pickle file containing derivation data, test data and transformation object "
                                  f"saved in: \n{os.path.join(os.path.dirname(train_file), pkl_fname)}")
            # copy sc_obj
            os.system(f"cp {sc_path} {os.path.dirname(train_file)}")
            glb.logger.log_string(f": copy of scaling_obj_*.pkl saved in {os.path.dirname(train_file)}")
    else:
        glb.logger.log_string(f": Loading derivation data, test data and transformation object from a pickle file")
        load_pkl_from = os.path.join(data_dir, "VLS_impute_train_test_" + utils.config['pred_type'] + "_" +
                                     utils.config['transformation'] + ".pkl")
        glb.logger.log_string(f"\t {load_pkl_from}")
        with open(load_pkl_from, "rb") as f:
            x_derivation, y_derivation, x_test, y_test = pickle.load(f)
        # copy sc_obj from data_dir to log_dir
        if sc_path is not None:
            os.system(f"cp {os.path.join(data_dir, 'scaling_obj_' + utils.config['transformation'] + '.pkl')} "
                      f"{glb.logger.log_dir}")
        if utils.config['transformation'] == 'ple_dt':  # load ple_dt object
            utils.config['n_features'] = x_derivation[0].shape[1]
            with open(os.path.join(data_dir, 'scaling_obj_' + utils.config['transformation'] + '.pkl'), "rb") as f:
                ple_dt = pickle.load(f)

    # data size
    n_samples_derivation, n_samples_test, n_feature = len(x_derivation), len(x_test), len(x_derivation[0][0])
    glb.logger.log_string(f"\n: Summary of derivation and test data: "
                          f"\n\t n_samples_derivation = {n_samples_derivation:,d}"
                          f"\n\t n_samples_test = {n_samples_test:,d} "
                          f"\n\t n_feature = {n_feature}")

    # validation data
    all_ids = np.arange(n_samples_derivation)  # split at study_id level
    np.random.shuffle(all_ids)  # shuffle before train/validation split
    if not val_file:
        # split derivation data into training and validation if no validation file provided.
        glb.logger.log_string(
            f": Splitting derivation data into training ({(1 - utils.config['validation_p']) * 100}%) and "
            f"validation ({utils.config['validation_p'] * 100}%)")
        num_validation = round(utils.config['validation_p'] * n_samples_derivation)
        # multiples of batch_size
        num_validation = (num_validation // utils.config['batch_size']) * utils.config['batch_size']
        train_ids = all_ids[:-num_validation]
        validation_ids = all_ids[-num_validation:]
        glb.logger.log_string(f"\t len(all_ids) = {len(all_ids):,d}")
        glb.logger.log_string(f"\t len(train_ids) = {len(train_ids):,d}")
        glb.logger.log_string(f"\t len(validation_ids) = {len(validation_ids):,d}")
        # prevalence of cases in train and validation data
        y_temp = y_derivation[train_ids]
        num_cases = np.sum([np.max(y_enc) for y_enc in y_temp])
        if utils.config['pred_type'] == 'seq2seq':
            y_temp = np.concatenate(y_temp).ravel()
        glb.logger.log_string(
            f"\t in training set: \n"
            f"\t\t num of encounters with outcome=1: {num_cases:,d} (out of {len(train_ids):,d})\n"
            f"\t\t num of timesteps with outcome=1: {sum(y_temp):,d} (out of {len(y_temp):,d})\n"
            f"\t\t prevalence of outcome=1: {sum(y_temp) / len(y_temp) * 100:.2f}%")
        y_temp = y_derivation[validation_ids]
        num_cases = np.sum([np.max(y_enc) for y_enc in y_temp])
        if utils.config['pred_type'] == 'seq2seq':
            y_temp = np.concatenate(y_temp).ravel()
        glb.logger.log_string(
            f"\t in validation set: \n"
            f"\t\t num of encounters with outcome=1: {num_cases:,d} (out of {len(validation_ids):,d})\n"
            f"\t\t num of timesteps with outcome=1: {sum(y_temp):,d} (out of {len(y_temp):,d})\n"
            f"\t\t prevalence of outcome=1: {sum(y_temp) / len(y_temp) * 100:.2f}%")
    else:
        # validation file provided. append validation data to derivation, but keep their ids separated
        # read val_file
        glb.logger.log_string(f"\n: Validation data")
        data_in = DataLoader(filenames=[val_file],
                             target="outcome",
                             loadcreate01=0,
                             sc_path=sc_path,
                             scaling=scaling)
        data_in.get_data_short(print_stats=False)
        df_input = data_in.data
        if utils.config['transformation'] == 'ple_dt':
            # apply transformation on test samples
            glb.logger.log_string(f": transforming validation set using PLE_DT")
            Xtrans = ple_dt.transform(df_input[[*data_in.pred_vars]].to_numpy())
            X = pd.concat([df_input[['study_id', 'outcome']],
                           pd.DataFrame(Xtrans)], axis=1)
            df_input = X.copy()
        x_val, y_val = prep_data_df_to_array(df_input)
        # append data
        x_derivation = np.append(x_derivation, x_val, axis=0)
        y_derivation = np.append(y_derivation, y_val, axis=0)
        # copy initial all_ids to train_ids
        train_ids = all_ids.copy()
        # update all_ids and validation_ids
        num_validation = len(x_val)
        validation_ids = np.arange(start=n_samples_derivation, stop=n_samples_derivation + num_validation)
        all_ids = np.arange(len(x_derivation))  # used later for final training using combination of train and val data
        np.random.shuffle(all_ids)

    # the class label imbalance
    if utils.config['add_weight']:
        y_train_tmp = y_derivation[train_ids]
        if utils.config['pred_type'] == 'seq2seq':
            # flatten y_train_tmp if seq2seq; otherwise it is already flat
            y_train_tmp = np.concatenate(y_train_tmp).ravel()
        neg, pos = np.bincount(y_train_tmp)
        weight_for_0 = (1 / neg) * ((neg + pos) / 2.0)
        weight_for_1 = (1 / pos) * ((neg + pos) / 2.0)
        class_weights = [weight_for_0, weight_for_1, 0.0]
        # class_weights = [1.0, 1.0, 0.0]
        glb.logger.log_string(f": sample weights: {class_weights}")
    else:
        # None if you opt out of adding class_weights
        class_weights = None

    # for seq2seq prediction Y has to be padded too
    pad_y = True if utils.config['pred_type'] == 'seq2seq' else False
    expand_dim = True if model_choice == 'mdl_tdcnn.py' else False
    # data has to be a generator or a (keras.utils.Sequence) instance for variable length sequences
    training_generator = VarLenSequence(x_derivation, y_derivation, list_IDs=train_ids,
                                        class_weights=class_weights,
                                        batch_size=utils.config['batch_size'], shuffle=True,
                                        padding=utils.config['padding'], pad_value=utils.config['mask_value'],
                                        expand_dim=expand_dim, pad_y=pad_y)
    validation_generator = VarLenSequence(x_derivation, y_derivation, list_IDs=validation_ids,
                                          class_weights=None if class_weights is None else [1.0, 1.0, 0.0],
                                          batch_size=utils.config['batch_size'], shuffle=True,
                                          padding=utils.config['padding'], pad_value=utils.config['mask_value'],
                                          expand_dim=expand_dim, pad_y=pad_y)
    # shuffle is off for test generator, to keep the order of samples for further AUC calculations
    test_generator = VarLenSequence(x_test, y_test, list_IDs=np.arange(len(x_test)),
                                    class_weights=None if class_weights is None else [1.0, 1.0, 0.0],
                                    batch_size=utils.config['batch_size'], shuffle=False,
                                    padding=utils.config['padding'], pad_value=utils.config['mask_value'],
                                    expand_dim=expand_dim, pad_y=pad_y)

    # ***** Set up search space
    glb.logger.log_string('\n---------- Setting up a search space')
    glb.logger.log_string('======================================\n')
    hp = HyperParameters()
    if utils.config['tuner_name'] == 'RandomSearch':
        tuner = RandomSearch(create_model, hyperparameters=hp, tune_new_entries=True,
                             objective=Objective("val_auroc", direction="max"),
                             # num_initial_points=5,  # only for Bayesian Optimization
                             max_trials=utils.config['max_trials'],  # comment for Hyperband
                             # max_epochs=MAX_TRIALS,   # uncomment for Hyperband
                             executions_per_trial=1, seed=utils.config['tuner_seed'],
                             directory=glb.logger.log_dir, project_name=utils.config['tuner_name'])
    elif utils.config['tuner_name'] == 'BayesianOptimization':
        tuner = BayesianOptimization(create_model, hyperparameters=hp, tune_new_entries=True,
                                     objective=Objective("val_auroc", direction="max"),
                                     num_initial_points=utils.config['num_initial_points'],  # for Bayesian Optimization
                                     max_trials=utils.config['max_trials'],  # comment for Hyperband
                                     # max_epochs=MAX_TRIALS,   # uncomment for Hyperband
                                     executions_per_trial=1, seed=utils.config['tuner_seed'],
                                     directory=glb.logger.log_dir, project_name=utils.config['tuner_name'])
    else:
        tuner = Hyperband(create_model, hyperparameters=hp, tune_new_entries=True,
                          objective=Objective("val_auroc", direction="max"),
                          # num_initial_points=5,  # for Bayesian Optimization
                          # max_trials=MAX_TRIALS,  # comment for Hyperband
                          max_epochs=utils.config['max_trials'],  # uncomment for Hyperband
                          executions_per_trial=1, seed=utils.config['tuner_seed'],
                          directory=glb.logger.log_dir, project_name=utils.config['tuner_name'])
    log_string_search_space_summary(tuner)
    # reload metrics if tuner is set to resume an unfinished tuning
    trials = tuner.oracle.trials
    if len(trials) > 0:
        glb.logger.log_string(f"\n :Resume tuning")
        glb.logger.log_string(f"id, score , step")
        glb.logger.log_string(f"----------------")
    for trial_id, trial in sorted(trials.items()):
        glb.logger.log_string(f"{trial_id}, {str(trial.score)[:6]}, {trial.best_step}")
        if trial.get_state().get('status') == 'COMPLETED':
            search_num[0] += 1
            log_trialids[trial_id] = [trial_id]
            log_metrics[trial_id] = [trial.best_step + 1]
            trial_log = trial.get_state().get('metrics').get('metrics')
            for v in trial_log.values():
                log_metrics[trial_id].append(v.get('observations')[0].get('value')[0])

    # ***** search for the optimal hyper-parameters
    glb.logger.log_string('\n---------- Tuning hyper-parameters')
    glb.logger.log_string('======================================\n')
    start_time = timeit.default_timer()
    # set up callback functions
    es_clbk = EarlyStopping(monitor='val_auroc', mode='max', patience=5, min_delta=0.005, verbose=1)
    # search hyperparameter space
    try:
        tuner.search(training_generator,
                     epochs=utils.config['max_epoch'],
                     # batch_size: Do not specify batch_size if data is in the form of generators
                     validation_data=validation_generator,
                     verbose=0,
                     callbacks=[es_clbk, CustomLoggingCallback()])
    except Exception as e:
        # in case of error
        glb.logger.log_string(f"{e}")
        # close the log file
        glb.logger.log_fclose()
        # terminate
        exit()
    # elapsed time
    elapsed = timeit.default_timer() - start_time
    glb.logger.log_string(f": Execution time of {utils.config['tuner_name']} search: {elapsed / 60} min")

    # ***** log the search summary
    glb.logger.log_string('\n---------- Search result summary:')
    glb.logger.log_string('======================================\n')
    log_metrics = pd.DataFrame.from_dict(log_metrics, orient='index',
                                         columns=['Epoch', 'loss', 'accuracy', 'auroc', 'auprc', 'val_loss',
                                                  'val_accuracy', 'val_auroc', 'val_auprc'])
    log_metrics = log_metrics.reset_index().rename(columns={'index': 'Trial_id'})
    log_metrics.index += 1  # shift index by 1, so that it starts at 1
    glb.logger.log_string(f": Log of monitored metrics for each run at the epoch with the best val_auroc:"
                          f" \n{log_metrics}\n")
    glb.logger.log_string(f": Best score = {tuner.oracle.get_best_trials(1)[0].score:.4f}")
    # store log_metrics in .csv file for future visualization
    log_metrics.to_csv(os.path.join(glb.logger.log_dir, 'log_metrics.csv'), float_format="%.4f", index=False)

    # ***** re-instantiate and retrain on the full dataset
    # For best performance, It is recommended to retrain your Model on the full dataset using the best hyperparameters
    # found during `search`
    glb.logger.log_string('\n---------- Re-instantiate and Train')
    glb.logger.log_string('======================================\n')
    best_hp = tuner.get_best_hyperparameters(1)[0]  # Returns the best hyperparameters, as determined by the objective
    model = tuner.hypermodel.build(best_hp)  # reinstantiate the (untrained) best model found during the search process
    best_epoch = log_metrics.loc[log_metrics['val_auroc'].idxmax(axis=0)]['Epoch']  # best epoch from hp search
    glb.logger.log_string(f": training for up to best_epoch = {best_epoch}")
    # the class label imbalance
    if utils.config['add_weight']:
        y_train_tmp = y_derivation[all_ids]
        if utils.config['pred_type'] == 'seq2seq':
            y_train_tmp = np.concatenate(y_train_tmp).ravel()
        neg, pos = np.bincount(y_train_tmp)
        weight_for_0 = (1 / neg) * ((neg + pos) / 2.0)
        weight_for_1 = (1 / pos) * ((neg + pos) / 2.0)
        class_weights = [weight_for_0, weight_for_1, 0.0]
        # class_weights = [1.0, 1.0, 0.0]
        glb.logger.log_string(f": sample weights: {class_weights}")
    else:
        class_weights = None
    # train on the full dataset
    training_generator = VarLenSequence(x_derivation, y_derivation, list_IDs=all_ids,
                                        class_weights=class_weights,
                                        batch_size=utils.config['batch_size'], shuffle=True,
                                        padding=utils.config['padding'], pad_value=utils.config['mask_value'],
                                        expand_dim=expand_dim, pad_y=pad_y)
    model.fit(training_generator,
              epochs=best_epoch,
              shuffle=False,
              verbose=2)
    model.save(os.path.join(glb.logger.log_dir, 'best_model.h5'))
    df_history = pd.DataFrame(model.history.history)
    glb.logger.log_string(f"\n: Training history:\n{df_history}")

    # ***** re-load the saved model and check the performance
    if model_choice == 'mdl_tcn.py':
        model = load_model(os.path.join(glb.logger.log_dir, 'best_model.h5'), custom_objects={'MyTCN': MyTCN})
    else:
        model = load_model(os.path.join(glb.logger.log_dir, 'best_model.h5'))
    glb.logger.log_string("\n: Evaluation of the saved model:")
    # open a strategy with as many devices as is visible to tf, in case batch size was an issue
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        glb.logger.log_string("\t on training data: " + ' - '.join(
            f"{m} : {v:.4f}" for m, v in zip(model.metrics_names, model.evaluate(training_generator, verbose=2))))
        glb.logger.log_string("\t on testing data: " + ' - '.join(
            f"{m} : {v:.4f}" for m, v in zip(model.metrics_names, model.evaluate(test_generator, verbose=2))))

    # ROC CI
    glb.logger.log_string(f"\n{datetime.datetime.today().strftime('%Y.%m.%d %H:%M:%S')}: "
                          f"Predicted probabilities of test dataset -> log_scores.csv")
    glb.logger.log_string(f"\t Change test generator to batch_size 1")
    test_generator = VarLenSequence(x_test, y_test, list_IDs=np.arange(len(x_test)),
                                    class_weights=None if class_weights is None else [1.0, 1.0, 0.0],
                                    batch_size=1, shuffle=False,
                                    padding=utils.config['padding'], pad_value=utils.config['mask_value'],
                                    expand_dim=expand_dim, pad_y=pad_y)
    if utils.config['pred_type'] == 'seq2one':
        y_pred = model.predict(test_generator).ravel()
    else:
        # model.predict cannot concat predictions with varying lengths. Solution: one sample at a time in a for loop
        # according to keras: for small numbers of inputs use __call__ function for faster execution
        y_pred = []
        glb.logger.log_string(f"\t test_generator.__len__()={test_generator.__len__()}")
        for i, enc_test in enumerate(test_generator):
            y_pred.append(np.reshape(model(enc_test[0], training=False).numpy(), -1))
            if (i % 10000) == 0:
                print(f"\t test encounter {i}")
        y_pred = np.concatenate(y_pred)
        y_test = np.concatenate(y_test).flatten()  # flatten y_test into 1D array of shape (n,)
    log_evaluate = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
    log_evaluate.to_csv(os.path.join(glb.logger.log_dir, 'log_scores.csv'), float_format="%.4f", index=False)
    out = utils.auc_ci_delong(os.path.join(glb.logger.log_dir, 'log_scores.csv'))
    glb.logger.log_string(datetime.datetime.today().strftime('%Y.%m.%d %H:%M:%S') + "\n" + out)

    # close the log file
    glb.logger.log_fclose()
