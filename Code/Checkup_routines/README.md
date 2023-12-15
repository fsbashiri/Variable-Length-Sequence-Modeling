# Checkup Routines

Within this section, you'll find several essential scripts designed to assess key aspects of the pipeline. These scripts focus on evaluating the causality of developed models, ensuring proper masking of padded sequences, and re-formatting the structured data. 

## 1. examine_datasets.py

The `examine_datasets.py` script serves a crucial role in preparing datasets for seamless integration into the main pipeline. Here's an overview of its functionalities:

- **Input and Diagnostics:** The script accepts one or more `.csv` files, conducts diagnostic tests, and ensures the integrity of variables and statistics. Minor adjustments are made to the tables to guarantee compatibility with the main pipeline.

- **Expected Dataframe Format:** The pipeline anticipates an input dataframe containing headers: ['study_id', 'time', 'outcome', *pred_vars*]. Any additional columns beyond these predictors must be dropped for proper functioning.

### Configurable Variables: 

Tailor the script as needed by editing the following variables:

- `data_dir`: Path to the directory in which data files are stored
- `filenames`: A list of [.csv] file names. All filenames must be stored in the same `data_dir` directory. The script will run on each file separately. Adjustments defined in the following configurable variables will be applied on all files.
- `cols_to_rename`: A list of table headers to be renamed in the `[(rename_from, rename_to)]` format. An empty list will instruct the script to skip header renaming. 
- `cols_to_drop`: A list of table headers that has to be dropped and not passed to the training process. The only acceptable non-predicting variables are: ['study_id', 'time', 'outcome']. An empty list will instruct the script to skip dropping columns.
- `time_higher_limit`: A dictionary that defines the long sequence trunction with the following keys:
  - "file_id": List of integers. Index of files from the `filenames` on which the truncation should apply.
  - "time_col": List of strings. Name of the time label in the `filenames[file_id]`.
  - "time_cap": List of floats. Within each file_id, time > time_cap will be dropped.   
- `save_path`: Path to the directory in which final data will be saved. Empty str if data should not be saved.

### How to Use:

To run the script, follow these steps:

1. Edit the `data_dir` variable within the script to specify the directory containing your `.csv` files.
2. Populate the `filenames` variable with the list of filenames to be examined individually. Note that column edits (e.g., renaming and dropping specified columns) will be applied universally.
3. Set the `time_higher_limit` variable to truncate long sequences, preventing Out-Of-Memory (OOM) issues during training. Truncation is determined by the `time_col` column and the `time_cap` value.
   
   - **Determining Time Cap Limit:** Before running the script, calculate the 95th and 99th percentiles of maxLOS at the encounter level using the following code snippet:
     ```python
     df_maxLOS = df_train.groupby('encounter_id').max()[['time_column']]
     print(f'train LOS quantiles: \n {df_maxLOS.quantile(q=[0.95, 0.99])}')
     ```
   - **Note:** Truncation is applicable only to the training and validation datasets. The test dataset remains untouched.


4. Edit the `save_path` variable as needed. The default path to save a compatible version of dataset is: `proj_dir/Data/Adult_Det_YYYY.mm.dd/`. Optionally, leave it empty (i.e., an empty string) if you do not need to save the final dataset.

5. Run the following command in a command terminal: 
   ```bash
   python Code/Checkup_routines/examine_datasets.py
   ```


## 2. examine_masking.py

The `examine_masking.py` script is designed to assess padding and masking within a model. It generates two test samples: one with `n_ts` time steps, and another identical sample with an additional time step filled with the specified `mask_value`. Both samples are then processed by a model, producing predictions that are thoroughly compared and examined. If the masking is working correctly the predictions for the masked and unmasked sequences are the same as well as the loss and other evaluating metrics. The script supports both **pre-padding** and **post-padding**. 

**Note:** Masking can only be tested when the `pred_type` is set to `seq2seq`.

### Configurable Variables: 

Tailor the script as needed by editing the following variables:

- `n_ts`: Integer. The length of the test sequence.
- `model_name`: String. Type of the model. Options: ['tcn', 'lstm_gru', 'tdcnn'].

### How to Use:

To execute the script, follow these steps:

1. Edit the `n_ts` and `model_name` variables in the code according to your requirements.
2. Run the following command in a command terminal: 
   ```bash
   python Code/Checkup_routines/examine_masking.py
   ```

## 3. examine_causality.py


The `examine_causality.py` script is developed to evaluate the causality aspect of a model. It generates two test samples: one with `n_ts` time steps, and another identical sample, but with the information from the last `n_mt` time step omitted. The key expectation is that predictions for both samples up to `(n_ts - n_mt)` time steps should be identical. This ensures that future measurements do not influence predictions from earlier time points, validating the causality of the model.

**Note:** Causality testing is applicable only when the `pred_type` is set to `seq2seq`.

### Configurable Variables: 

Tailor the script as needed by editing the following variables:

- `n_ts`: Integer. The length of the test sequence.
- `n_mt`: Integer. The number of time steps in the future that one test sample has more than the other.
- `model_name`: String. Type of the model. Options: ['tcn', 'lstm_gru', 'tdcnn']

### How to Use:

To execute the script, follow these steps:

1. Edit the `n_ts`, `n_mt`, and `model_name` variables in the code according to your specifications.
2. Run the following command in a command terminal: 
   ```bash
   python Code/Checkup_routines/examine_causality.py
   ```

## 4. examine_causality_ext.py

The `examine_causality_ext.py` script is an extended version of the 'examine_causality.py' script. It addresses a specific scenario where predictions made by a TCN model exhibit variations at a high decimal. This extended script performs causality tests iteratively with different numbers of time steps, assessing the precision of predictions up to a specified decimal value.

### Configurable Variables: 

Tailor the script as needed by editing the following variables:

- `n_patients`: Integer. Length of test sequence at axis=0.
- `n_ts`: List of integer. The experiment repeats for the length of the list. Each element of the list shows the length of the longest test sequence. For example, for `n_ts[0]` the shape of the longest test sequence is: `(n_patients, n_ts[0], n_features)`. The shorter sequence will be created by deleting one time step from the longer sequence and repeats the experiment until one time step is left. 
- `model_name`: String. Type of the model. Options: ['tcn', 'lstm_gru', 'tdcnn']

### How to Use:

To execute the script, follow these steps:

1. Edit the `test_ts`, `n_patients`, and `model_name` variables in the code as needed.
2. Run the following command in a command terminal: 
   ```bash
   python Code/Checkup_routines/examine_causality_ext.py
   ```
   
