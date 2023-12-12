"""
Project: Variable Length Sequence Modeling
Author: Azi Bashiri
Last Modified: April 2023
Description: A script to evaluate the performance of one stored model in a test dataset.
            This script loads a model in .h5 format, loads test dataset from a .pkl object, calculates predicted
            probabilities, stores them along with true labels in a log_scores.csv file, and calculates 95% AUC_CI.
            The location of the saved model is provided via utils.config variable (i.e., log_path/log_folder). Model
            type and the path to the data object are provided to the code via input arguments. The data must had been
            pre-processed and stored in .pkl format.
Note: if the number of processes is too high and you are receiving an error message that states:
        "OpenBLAS blas_thread_init: pthread_create failed for ..." Here are two possible ways to fix:
        1. ask the admin to increase the "max user processes" value for your account (type in a terminal: ulimit -a)
        2. deal with the problem by changing the max number of threads in python by following these steps:
            a. uncomment line 49 (i.e, os.environ['OPENBLAS_NUM_THREADS'] = '30')
            b. check this link that shows what value should be used: https://stackoverflow.com/a/67977275
"""
# import packages
import datetime
import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import multiprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # GitHub tensorflow issue #59779
cwd = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.abspath(os.path.join(cwd, os.pardir))
sys.path.append(proj_dir)
sys.path.append(os.path.abspath(os.path.join(proj_dir, os.pardir, 'rtdl')))
from Code.VL010_my_seq_class import VarLenSequence
import Code.VL099_utils as utils
import Code.VL011_globals as glb
from Code.Models.mdl_tcn import MyTCN
# run on CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{-1}"
import tensorflow as tf
from keras.models import load_model


# random seed
np.random.seed(777)
processes = 10  # number of processes in a pool
print_every = 1000  # debugging. print status every print_every indexes
# os.environ['OPENBLAS_NUM_THREADS'] = '30'  # uncomment if max user processes is not enough

# Do not squeeze when printing
np.set_printoptions(threshold=sys.maxsize)  # print numpy arrays completely
pd.set_option("display.max_rows", None, "display.max_columns", None)  # print pandas dataframes completely


def worker_init(worker_model_path, worker_model_choice):
    """
    Initialize all workers with loading a worker_model.
    In spawn method variables are not inherited by child processes. Also, a Keras model cannot be passed to the worker
    function through arguments because it cannot be pickled/unpickled. So, we load an individual model for each process
    in an initializer function.
    :param worker_model_path: path to saved model (e.g., path/to/best_model.h5)
    :param worker_model_choice: model choice. options: mdl_tcn.py, mdl_lstm_gru.py, mdl_tdcnn.py
    :return: None. worker model is a global variable
    """
    # global worker_model
    global worker_model
    print(f"{datetime.datetime.today().strftime('%Y.%m.%d %H:%M:%S')} "
          f"Initialize worker_init for {multiprocessing.current_process().name}")
    if worker_model_choice == 'mdl_tcn.py':
        worker_model = load_model(worker_model_path, custom_objects={'MyTCN': MyTCN})
    else:
        worker_model = load_model(worker_model_path)
    print(f"{datetime.datetime.today().strftime('%Y.%m.%d %H:%M:%S')} "
          f"Model loaded for worker {multiprocessing.current_process().name}")
    return


def worker_func(x):
    """
    worker function. called by each process at every iteration
    :param x: input argument (encounter_x, encounter_id)
    :return: numpy array of predicted probabilities in 1D shape (i.e., reshaped to (-1,))
    """
    # unpack input arg
    encounter_x, idx = x
    if idx % print_every == 0:
        print(f"{datetime.datetime.today().strftime('%Y.%m.%d %H:%M:%S')} "
              f"i: {idx}, pid: {os.getpid()}, process_name: {multiprocessing.current_process().name}, "
              f"input_shape: {encounter_x.shape}")
    # for small numbers of inputs, model(x) is faster than model.predict(x)
    worker_model_prediction = worker_model(encounter_x, training=False).numpy()
    if idx % print_every == 0:
        print(f"{datetime.datetime.today().strftime('%Y.%m.%d %H:%M:%S')} "
              f"Model prediction complete for index {idx}")
    return np.reshape(worker_model_prediction, -1)


def start_logging():
    # ***** Start logging *****
    glb.init_logging(b_log_txt=True, log_name="log_evaluate.txt",
                     log_path=utils.config['log_path'],
                     log_folder=utils.config['log_folder'])
    # copy files for backup.
    os.system(f"cp {os.path.join(proj_dir, 'Code', 'VL040_evaluate.py')} {glb.logger.log_dir}")  # bkp of script
    glb.logger.log_string('Python %s on %s' % (sys.version, sys.platform))
    glb.logger.log_string(glb.logger.__str__())
    return


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluating a trained model in sequential data with varying lengths",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model",
                        default="lstm_gru", type=str,
                        help="Model of choice. Default: 'lstm_gru' \n"
                             "Options: \n\t"
                             "'tcn'= temporal convolution network \n\t"
                             "'lstm_gru'= multi-layer LSTM and GRU \n\t"
                             "'tdcnn'= time distributed wrapper with CNN")
    parser.add_argument("-d", "--data_dir",
                        default="", type=str,
                        help="Path to data directory for fast loading data from a pickle object. Default: ''")
    parser.add_argument("-f", "--model_name",
                        default="best_model.h5", type=str,
                        help="File name of a model to load. Default: 'best_model.h5'")
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    if np.isin(args.model, ['tcn', 'tdcnn', 'lstm_gru']):
        model_choice = "mdl_" + args.model + ".py"
    else:
        raise AssertionError(f"Unknown model of choice: {args.model}")
    if os.path.isdir(args.data_dir):  # data_dir: an existing directory. It cannot be an empty string
        data_dir = args.data_dir
    else:
        raise NotADirectoryError(f"data_dir = {args.data_dir}")
    if not args.model_name:
        raise ValueError(f"Argument model_name cannot be empty.")
    elif args.model_name[-3:].lower() != '.h5':
        raise ValueError(f"Argument model_name must have extension '.h5'")
    else:
        model_name = args.model_name

    # ***** Start logging *****
    start_logging()
    glb.logger.log_string(f": config: \n{utils.config}")

    # ***** load model
    glb.logger.log_string('\n---------- Loading model')
    model_path = os.path.join(utils.config['log_path'], utils.config['log_folder'], model_name)
    # check if model exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    # load model
    if model_choice == 'mdl_tcn.py':
        model = load_model(model_path, custom_objects={'MyTCN': MyTCN})
    else:
        model = load_model(model_path)
    glb.logger.log_string(f": Model loaded from: \n{model_path}")
    # model summary
    model.summary(print_fn=lambda x: glb.logger.log_string(x))

    # ***** get the data
    glb.logger.log_string('\n---------- Loading data')
    data_path = os.path.join(data_dir, "VLS_impute_train_test_" + utils.config['pred_type'] + "_" +
                             utils.config['transformation'] + ".pkl")
    with open(data_path, "rb") as f:
        _, _, x_test, y_test = pickle.load(f)
    glb.logger.log_string(f": Data loaded from: \n{data_path}")
    # update n_features if PLE_DT
    if utils.config['transformation'] == 'ple_dt':  # load ple_dt object
        utils.config['n_features'] = x_test[0].shape[1]
        glb.logger.log_string(f": n_features updated to {utils.config['n_features']}")
    # print info on data
    n_samples_test = len(x_test)
    glb.logger.log_string(f"\t n_samples_test = {n_samples_test}")

    # ***** test generator
    glb.logger.log_string('\n---------- Creating test generator')
    # class weights for test samples are either 1.0 or None
    class_weights = [1.0, 1.0, 0.0] if utils.config['add_weight'] else None
    # for seq2seq prediction Y has to be padded too
    pad_y = True if utils.config['pred_type'] == 'seq2seq' else False
    # only tdcnn model needs expand_dim to be true
    expand_dim = True if model_choice == 'mdl_tdcnn.py' else False
    # data has to be a generator or a (keras.utils.Sequence) instance for variable length sequences
    test_generator = VarLenSequence(x_test, y_test, list_IDs=np.arange(n_samples_test),
                                    class_weights=class_weights,
                                    batch_size=1,
                                    shuffle=False,
                                    padding=utils.config['padding'],
                                    pad_value=utils.config['mask_value'],
                                    expand_dim=expand_dim,
                                    pad_y=pad_y)

    # ***** start making predictions
    glb.logger.log_string('\n---------- Start making predictions')
    glb.logger.log_string(f": Predicted probabilities in test dataset -> log_scores.csv")

    # multiprocessing in CPU
    with tf.device('CPU'):
        st = time.time()
        glb.logger.log_string(f"{datetime.datetime.today().strftime('%Y.%m.%d %H:%M:%S')}: Creating embedded_params")
        embedded_params = ((encounter_xy[0], i) for i, encounter_xy in enumerate(test_generator))
        glb.logger.log_string(f"{datetime.datetime.today().strftime('%Y.%m.%d %H:%M:%S')}: "
                              f"Creating Pool with {processes} processes")
        with multiprocessing.get_context("spawn").Pool(processes=processes,
                                                       initializer=worker_init,
                                                       initargs=(model_path, model_choice,)) as pool:
            # predicted probability
            y_pred = pool.map(worker_func, embedded_params)
        glb.logger.log_string(f"{datetime.datetime.today().strftime('%Y.%m.%d %H:%M:%S')}: Done. "
                              f"len(y_pred) = {len(y_pred)}")
        et = time.time()
        glb.logger.log_string(f": Elapsed time: {((et - st) / 60):.2f} minutes.")
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_test).flatten()
        # save y_pred and y_true in a CSV file
        log_evaluate = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        log_evaluate.to_csv(os.path.join(glb.logger.log_dir, 'log_scores_mp.csv'), float_format="%.4f", index=False)
        # ROC CI (DeLong method)
        out = utils.auc_ci_delong(os.path.join(glb.logger.log_dir, 'log_scores_mp.csv'))
        glb.logger.log_string(datetime.datetime.today().strftime('%Y.%m.%d %H:%M:%S') + "\n" + out)

    # close the log file
    glb.logger.log_fclose()
