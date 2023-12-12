"""
Author: Azi Bashiri
Created: Oct. 2022
Last Modified: Nov. 2022
Description: A python script to examine datasets. It takes in one or more .csv files, runs diagnostics tests, ensures
                all variables and statistics look good, and make minor changes to the table so that they will be
                compatible with the code in the main pipeline.
Usage:  1. Review definitions right below import packages and modify them as needed
        2. In a terminal change directory to the project folder.
            Then type `python Code/Checkup_routines/examine_datasets.py` and press Enter.

"""
# import packages
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
cwd = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(proj_dir)
import Code.VL011_globals as glb
from Code.Data_loader.data_loader import DataLoader

# definitions - dataset specific config
data_dir = "/Path/to/Data/Directory"
filenames = ["filename_0.csv",
             "filename_1.csv"]
# rename table headers. empty list if none. Format: [(rename_from, rename_to)]
cols_to_rename = [('encounter_id', 'study_id'),
                  ('outcome24hr_det', 'outcome')]
# cols to drop. empty list if none
cols_to_drop = ['patient_id', 'loc_ed', 'loc_ward', 'loc_icu']  # Adult_Det dataset
# cols_to_drop = ['patient_id']  # AKI and Sepsis dataset
# higher limit on time. Empty list for each field if not interested in applying it to any dataset
time_higher_limit = {"file_id": [0],  # index of file in the list of filenames
                     "time_col": ["hours_since_admit"],  # time label in the filename
                     "time_cap": []}  # truncate LOS at 99th percentile.
# save_path. empty str if data should not be saved (comment out from + sign thereafter)
save_path = "" + os.path.join(proj_dir, 'Data', 'Adult_Det_' + datetime.today().strftime("%Y.%m.%d"))

# Do not squeeze when printing
np.set_printoptions(threshold=sys.maxsize)  # print numpy arrays completely
pd.set_option("display.max_rows", None, "display.max_columns", None)  # print pandas dataframes completely

if __name__ == '__main__':
    # start logging
    glb.init_logging(b_log_txt=True, log_name="log_dataset_checkup.txt")
    glb.logger.log_string(f"Python version {sys.version.split(sep=' ')[0]} on {sys.platform} platform")
    glb.logger.log_string(glb.logger.__str__())

    # copy this file to log_dir
    os.system(f"cp {os.path.join(cwd, 'examine_datasets.py')} {glb.logger.log_dir}")

    # add data_dir to filenames
    filenames = [os.path.join(data_dir, filename) for filename in filenames]

    # loop over filenames
    for file_id, filename in enumerate(filenames):
        glb.logger.log_string(f"\n----- start checkup routine \n===================================\n")
        # data loader object
        data = DataLoader(filenames=[filename])
        data.read_csv_files()

        # run diagnostics
        # rename column names
        if len(cols_to_rename) > 0:
            data.data.rename(dict(cols_to_rename), axis=1, inplace=True, errors='raise')
            glb.logger.log_string(f": columns renamed: {cols_to_rename}")

        # drop columns
        data.drop_extra_columns(cols_to_drop=cols_to_drop)
        glb.logger.log_string(f": columns dropped: {cols_to_drop}")

        # set a higher limit on time column
        for i in range(len(time_higher_limit.get('file_id'))):
            if time_higher_limit.get('file_id')[i] == file_id:
                data.data = \
                    data.data[data.data[time_higher_limit.get('time_col')[i]] <= time_higher_limit.get('time_cap')[i]]
                glb.logger.log_string(f": time column capped: {time_higher_limit.get('time_col')[i]} <= "
                                      f"{time_higher_limit.get('time_cap')[i]}")

        # number of unique study_ids
        if np.isin('study_id', data.data.columns.to_list()):
            glb.logger.log_string(f": number of unique study_id: {len(data.data['study_id'].unique())}")
        else:
            glb.logger.log_string(f"!! study_id not found.")
            raise AssertionError(f"Possibly study_id is named something else in the dataset. "
                                 f"Use cols_to_rename to rename columns, e.g., [('encounter_id', 'study_id')].")

        # max and min length of tseq
        if np.isin('outcome', data.data.columns.to_list()):
            glb.logger.log_string(f": min_tseq: {data.data.groupby(['study_id']).count()[data.target].min()}")
            glb.logger.log_string(f": max_tseq: {data.data.groupby(['study_id']).count()[data.target].max()}")
        else:
            glb.logger.log_string(f"!! outcome not found.")
            raise AssertionError(f"Possibly outcome is named something else in the dataset. "
                                 f"Use cols_to_rename to rename columns, e.g., [('outcome24hr', 'outcome')].")
        # list pred_vars
        data.update_pred_vars(inplace=True)
        glb.logger.log_string(f": predictor variables: total = {len(data.pred_vars)} \n{data.pred_vars}")

        # check if data is imputed (is there any NA?)
        # number of NAs for each column (variable)
        num_nas = data.data.isna().sum(axis=0)  # Pandas series
        nonzero_na_index = num_nas.index[num_nas > 0].to_list()
        glb.logger.log_string(f": columns with NA: total = {len(nonzero_na_index)} {nonzero_na_index}")

        # statistics of columns
        glb.logger.log_string(f": statistics of data: \n{data.data.describe()}")

        # save data in save_path, if not empty
        if len(save_path) > 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            fname_out = filename.split(sep="/")[-1][:-4] + "_checkup_" + datetime.today().strftime("%Y.%m.%d") + ".csv"
            data.data.to_csv(os.path.join(save_path, fname_out), index=False)
            glb.logger.log_string(f": dataset saved in: {os.path.join(save_path, fname_out)}")

    # close the log file
    glb.logger.log_fclose()
