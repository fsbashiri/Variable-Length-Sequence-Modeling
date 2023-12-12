"""
Project: Variable Length Sequence Modeling
Branch:
Author: Azi Bashiri
Last Modified: Mar. 2023
Description:
Convert datasets into a compatible format prior to using the data_loader with the Checkup_routines/examine_datasets.py
script. Acceptable input data is a .csv file that contains only the following columns:
    - study_id: representative of patient/encounter id
    - time: representative of hour block
    - outcome: outcome column
    - predicting variables: any column other than those mentioned above is a predicting variable
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
cwd = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(proj_dir)
import Code.VL011_globals as glb


class DataLoader:
    def __init__(self, filenames, target="outcome", loadcreate01=0, sc_path='', scaling='none'):
        self.filenames = filenames  # list of input filenames, e.g. [fname1, fname2]
        self.target = target        # name of target column
        self.loadcreate01 = loadcreate01  # binary: load scaler obj. from sc_path if loadcreate01==0; create otherwise
        self.sc_path = sc_path      # path to scaler object
        self.scaling = scaling      # options: 'none', 'min_max', 'mean_std'

        self.non_pred_cols = ['study_id', 'time', self.target]  # list of non-predictor variables
        self.pred_vars = []      # list of predictor variables (i.e., features)
        self.data = pd.DataFrame()  # dataframe
        self.x = []     # 2D numpy array of data - predictors
        self.y = []     # 1D numpy array of outcome - one value for every study_id
        self.max_tseq = 0   # maximum length of a sequence

        # check sc_path
        if self.sc_path == '':
            self.sc_path = os.path.join(proj_dir, 'Output', 'scaling_obj.pkl')
            glb.logger.log_string(f"sc_path was left empty. It is set to: {self.sc_path}")
        else:
            if self.loadcreate01 == 1:
                # if it's going to create an sc object, the parent folder must exist
                if not os.path.exists(os.path.dirname(self.sc_path)):
                    raise AssertionError(f"Parent directory to store a scaling object was not found in: {self.sc_path}")
            else:
                # if it's going to load an sc object, the file must exist
                if not os.path.exists(self.sc_path):
                    raise AssertionError(f"Scaling object was not found in sc_path: {self.sc_path}")

    def drop_extra_columns(self, cols_to_drop=None):
        if cols_to_drop is None:  # solution to not using mutable default values
            cols_to_drop = []
        self.data.drop(labels=cols_to_drop, axis=1, errors='ignore', inplace=True)  # drop extra columns
        return

    def fill_backward(self, col_name='loc_cat'):
        # col_name is valid
        if col_name not in self.data.columns.to_list():
            raise AssertionError(f"col_name {col_name} doesn't exist.")
        # fill backward if has NA
        if self.data[col_name].isna().any():
            self.data[col_name].fillna(method='bfill', inplace=True)
        return

    def encode_loc_cat(self):
        # one-hot encoding of loc_cat variable (Ward:1, ICU:2, OR:3, ER:4, Inv/Diag/Other:6)
        unique_loc_cat = self.data['loc_cat'].unique().tolist()
        if type(unique_loc_cat[0]) == int:
            # if loc_cat is encoded to numeric values
            self.data = pd.get_dummies(self.data, columns=['loc_cat'], prefix_sep='.')
        elif type(unique_loc_cat[0]) == str:
            # if loc_cat contains strings
            loc_cat_drop_list = ['loc_cat_' + item for item in
                                 set(unique_loc_cat) - {'WARD', 'ER', 'OR', 'ICU', 'INVT/DIAG/OTHER'}]
            self.data = pd.get_dummies(self.data, columns=['loc_cat']).drop(loc_cat_drop_list, axis=1)
            self.data.rename({'loc_cat_WARD': 'loc_cat.1', 'loc_cat_ICU': 'loc_cat.2',
                              'loc_cat_OR': 'loc_cat.3', 'loc_cat_ER': 'loc_cat.4',
                              'loc_cat_INVT/DIAG/OTHER': 'loc_cat.6'}, axis=1, inplace=True)
        else:
            raise AssertionError(f"Unknown loc_cat data type. It has to be either 'int' or 'str'.")
        return

    def scale_data(self, verbose=0):
        if self.scaling is None:
            # no scaling performed
            glb.logger.log_string(f": scaling is set to '{self.scaling}'. No scaling was performed.")
        elif self.scaling.lower() == 'min_max':
            glb.logger.log_string(f": scaling data -> Normalizing to min=0, max=1")
            if self.loadcreate01 == 1:  # create scaler object
                scaler = MinMaxScaler().fit(self.data[self.pred_vars])
                with open(self.sc_path, 'wb') as output:
                    pickle.dump(scaler, output)
                    glb.logger.log_string(f"\t Scaler object is saved at: {self.sc_path}")
            else:  # load scaler object otherwise
                scaler = pickle.load(open(self.sc_path, 'rb'))
                glb.logger.log_string(f"\t Scaler object is loaded from: {self.sc_path}")
            self.data[self.pred_vars] = scaler.transform(self.data[self.pred_vars])
            if verbose == 1:
                # print out describe of data
                glb.logger.log_string(f": Descriptive statistics of data: \n{self.data.describe()}")
        elif self.scaling.lower() == 'mean_std':  # standardization (mean=0, std=1)
            glb.logger.log_string(f": scaling data -> Standardize to mean=0, std=1")
            # detect non-binary columns
            non_bool_col = [col for col in self.pred_vars
                            if not np.isin(self.data[col].dropna().unique(), [0, 1]).all()]
            if self.loadcreate01 == 1:  # create scaler object
                scaler = StandardScaler().fit(self.data[non_bool_col])
                with open(self.sc_path, 'wb') as output:
                    pickle.dump(scaler, output)
                    glb.logger.log_string(f"\t Scaler object is saved at: {self.sc_path}")
            else:  # load scaler object otherwise
                scaler = pickle.load(open(self.sc_path, 'rb'))
                glb.logger.log_string(f"\t Scaler object is loaded from: {self.sc_path}")
            # standardize only on non-binary columns
            self.data[non_bool_col] = scaler.transform(self.data[non_bool_col])
            if verbose == 1:
                # print out describe of data
                glb.logger.log_string(f": Descriptive statistics of data: \n{self.data.describe()}")
        else:
            raise UserWarning(f"Incorrect scaling. Scaling can be 'none', 'min_max', or 'mean_std'.")
        return

    def read_csv_files(self):
        # clear data attribute
        self.data = pd.DataFrame()
        # read and concat
        for filename in self.filenames:
            glb.logger.log_string(f": Reading file: {filename}")
            df_tmp = pd.read_csv(filename)
            glb.logger.log_string(f"\t Shape of data: {df_tmp.shape}")
            self.data = pd.concat([self.data, df_tmp], ignore_index=True).reset_index(drop=True)
        if len(self.filenames) > 1:
            # print out shape of data if more than 1 file was concatenated
            glb.logger.log_string(f": Shape of dataframe: {self.data.shape}")
        return

    def update_pred_vars(self, inplace=False):
        out = [item for item in self.data.columns.to_list() if item not in set(self.non_pred_cols)]
        if inplace is True:
            self.pred_vars = out
            return
        else:
            return out

    def get_data_short(self, print_stats=False):
        """
        A short version of get_data method. This method reads input files, concatenate them along the axis=0, encodes
         'loc_cat' variable with one-hot encoding, drops non-predictor variables from data, scales data, and finally
         provides three important attributes:
         self.data: Dataframe with ['encounter_id', target, feature data]
         self.x: 2D numpy array of scaled feature data in the shape of (len of data, len of features)
         self.y: 1D numpy array of outcome in the shape of (num patient,)
         :param print_stats: print out statistics of predictor variables if true. Default: False
        """
        # read filenames into self.data
        self.read_csv_files()

        # length of time sequence
        self.max_tseq = self.data.groupby(['study_id']).count()[self.target].max()

        # encode loc_cat
        if np.isin('loc_cat', self.data.columns.to_list()):
            # only encode if it is not encoded yet
            self.encode_loc_cat()

        # list names of predictor variables after encoding loc_cat
        self.update_pred_vars(inplace=True)

        # Scaling the data - only pred_vars columns
        self.scale_data(verbose=0)

        # convert to numpy
        self.x = self.data[self.pred_vars].to_numpy()
        self.y = self.data.groupby('study_id').max()[self.target].to_numpy()

        # Log info about training data
        glb.logger.log_string('\n: Summary of data:')
        glb.logger.log_string(f"\t x shape: {self.x.shape}")
        glb.logger.log_string(f"\t y shape (max outcome per encounter): {self.y.shape}")
        glb.logger.log_string(f"\t number of unique study_id: {len(self.data['study_id'].unique())}")
        glb.logger.log_string(f"\t maximum number of time steps: {self.max_tseq}")
        glb.logger.log_string(f"\t outcome prevalence (at encounter level): "
                              f"{self.target}==1 ({self.y.sum()}) - "
                              f"{self.target}==0 ({self.y.shape[0] - self.y.sum()})")
        glb.logger.log_string(f"\t outcome prevalence (at time step level): "
                              f"{self.target}==1 ({self.data[self.target].sum()}) - "
                              f"{self.target}==0 ({self.data.shape[0] - self.data[self.target].sum()})")
        glb.logger.log_string(f"\t Predictor variables: {self.pred_vars}")
        if print_stats:
            glb.logger.log_string(f"\t description of data: \n{self.data.describe()}")
        return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(f"cwd: {cwd}")
    print(f"proj_dir: {proj_dir}")
    glb.init_logging(b_log_txt=True, log_name="my_log.txt")
    gs_data = os.path.join(proj_dir, "Data", "filename.csv")
    my_data = DataLoader(filenames=[gs_data],
                         target="outcome",
                         loadcreate01=1,
                         sc_path=os.path.join(glb.logger.log_dir, 'scaling_object.pkl'),
                         scaling='min_max')
    my_data.get_data_short()
    glb.logger.log_fclose()
