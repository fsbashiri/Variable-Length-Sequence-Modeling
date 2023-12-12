"""
Author: Azi Bashiri
Created: Oct. 2022
Last Modified: Oct. 2022
Description: A python script to run an extensive test for causality of a model. This script is an extended version of
                the 'examine_causality.py' script, after we found some predictions made by a TCN model were different
                at a high decimal.
                Causality can only be tested when pred_type is set to seq2seq.
Usage:  1. Review definitions right below import packages and modify them as needed
        2. In a terminal change directory to the project folder.
            Then type `python Code/Checkup_routines/examine_causality_ext.py` and press Enter.

"""
# import packages
import os
import sys
import math
import numpy as np
import pandas as pd
from keras_tuner.engine.hyperparameters import HyperParameters
cwd = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.join(cwd, os.pardir, os.pardir)
sys.path.append(proj_dir)
import Code.VL011_globals as glb
import Code.VL099_utils as utils

# definitions
test_ts = [5, 10, 15, 20]  # list of max_len_seq for the longer sequence samples
n_patients = 5
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


def truncate(numbers, digits):
    """
    truncate float numbers up to 'digits' decimal places
    :param numbers: numpy array of float number
    :param digits: int value
    :return: list of truncated numbers
    """
    numbers = list(numbers)
    stepper = 10 ** digits
    result = [math.trunc(stepper * number) / stepper for number in numbers]
    return result


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
    os.system(f"cp {os.path.join(cwd, 'examine_causality_ext.py')} {glb.logger.log_dir}")

    # create model
    hps = HyperParameters()
    my_model = create_model(hps)

    comp_result = {
        'n_patient': [],
        'x1_len': [],
        'x2_len': [],
        'missed_sample': [],
        'missed_time': [],
        'decimal_point': []
    }  # empty dict
    for _, n_ts in enumerate(test_ts):
        test_mt = np.arange(1, n_ts)
        for _, n_mt in reversed(list(enumerate(test_mt))):
            x2_seq_len = n_ts - n_mt

            # create x1, x2
            x1 = np.random.rand(n_patients, n_ts, utils.config['n_features'])
            x2 = np.delete(x1, np.s_[-n_mt:], axis=1)
            if model_name == 'tdcnn':
                # input shape for tdcnn is [None, None, n_features, 1, 1]
                x1 = np.expand_dims(np.expand_dims(x1, axis=-1), axis=-1)
                x2 = np.expand_dims(np.expand_dims(x2, axis=-1), axis=-1)

            # predictions
            y1_pred = my_model(x1).numpy()
            y2_pred = my_model(x2).numpy()
            # assess sequence predictions
            for id in range(n_patients):
                if np.all(y1_pred[id, :-n_mt, :] == y2_pred[id, :, :]):
                    # test passed
                    glb.logger.log_string(f"CAUSAL predictions. x1_len:{n_ts}, x2_len:{x2_seq_len}, sample:{id}")
                else:
                    dc = 1  # number of decimal places
                    while dc < 11:  # compare up to 10 decimal places
                        # compare y1_pred and y2_pred upto 'dc' decimal places
                        y1_trunc = truncate(y1_pred[id, :-n_mt, 0], dc)
                        y2_trunc = truncate(y2_pred[id, :, 0], dc)
                        indices = np.where(np.not_equal(y1_trunc, y2_trunc))[0]
                        if len(indices) == 0:
                            # equal values upto 'dc' decimal points
                            dc += 1
                        else:
                            # test failed
                            comp_result['n_patient'].append(n_patients)
                            comp_result['x1_len'].append(n_ts)
                            comp_result['x2_len'].append(x2_seq_len)
                            comp_result['missed_sample'].append(id+1)
                            comp_result['missed_time'].append(indices)
                            comp_result['decimal_point'].append(dc)
                            break

    # save the result
    comp_result = pd.DataFrame.from_dict(comp_result)
    glb.logger.log_string(f"Comparison reslut: \n{comp_result}")

    # close the log file
    glb.logger.log_fclose()
