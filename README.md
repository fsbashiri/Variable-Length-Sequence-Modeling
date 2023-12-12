# Variable-Length Sequence Modeling

Welcome to the repository dedicated to training deep learning models for variable-length time series data extracted from electronic health records, with a focus on predicting clinical outcomes. If you're interested in understanding the details of this work, you can explore the accompanying paper [here-TBD](link-to-paper).


## Requirements

- Python3
- For python dependencies, see [requirements.txt](https://git.doit.wisc.edu/smph-public/dom/uw-icu-data-science-lab-public/variable-length-sequence-modeling/-/blob/main/requirements.txt)
- R libraries:
  - rms
  - pROC
  

## Usage Guide

Follow the following steps for running the code on your data:

**1. Clone the Repository**

After cloning the repository, navigate to the project folder using the command line:
```bash
cd path/to/project/directory
```

**2. Install Dependencies**

Ensure you have all the required dependencies by executing the following command:
```bash
pip install -r requirements.txt
```

**3. Dataset Formatting**

Convert your dataset into a format compatible with the general pipeline:
```bash
python Code/Checkup_routines/examine_datasets.py
```

For detailed information and options, refer to the [Checkup Routines README](https://git.doit.wisc.edu/smph-public/dom/uw-icu-data-science-lab-public/variable-length-sequence-modeling/-/blob/main/Code/Checkup_routines/README.md).

**4. Run the Pipeline**

Execute the pipeline by running the following command:
```bash
python Code/VL020_train.py -m Model -t path/to/train/csv/file -e path/to/evaluation/csv/file -v path/to/validation/csv/file
```

Explore additional options in the [Important Notes](https://git.doit.wisc.edu/smph-public/dom/uw-icu-data-science-lab-public/variable-length-sequence-modeling#important-notes) section for a comprehensive understanding of available functionalities.


## Important Notes

Here are important notes to consider before running the code:

1. For detailed information about input arguments, type the following command in the command line:
   ```bash
   python Code/VL020_train.py -h
   ```

2. Before running the code, review and edit the **config** dictionary defined in `VL099_utils.py` as needed.

3. **Training and Test Datasets:**
   - Provide datasets in two ways:
     1. Direct the code to [.csv] files using `-t path/to/train/csv/file` and `-e path/to/evaluation/csv/file` arguments.
     2. Direct the code to [.pkl] objects containing derivation and evaluation data objects. Ensure that **config['save_pkl']** is set to True during the previous saving.

4. **Validation Dataset:**
   - Provide the validation dataset in two ways:
     1. Specify a [.csv] file using the `-v path/to/valid/csv/file` input argument.
     2. Leave the `-v` input argument empty to enable the code to split the training dataset into training and validation datasets based on the **validation_p** key in the **config** dictionary.

5. **Data Pre-processing:**
   - Select the data pre-processing method in the **config** dictionary from the list below: 
     1. "norm" -> Normalization: Transforming data to the range [0, 1]. 
     2. "std" -> Standardization: Adjusting data to have zero mean and unit variance.
     3. "ple-dt" -> Piece-wise Linear Encoding with Decision Trees (PLE-DT): A specialized encoding technique.

6. **Models:**
   - Select the model architecture you want to train with the `-m` input argument from the list below:
     1. "lstm" -> LSTM/GRU 
     2. "tdcnn" -> TDW-CNN
     3. "tcn" -> TCN

7. **Resuming Training:**
   - Resume an unfinished training pipeline from the last stored checkpoint by directing the code to the log directory using the **config['log_path']** and **config['log_folder']** keys.

8. **RTDL Remote Repository:**
   - The RTDL remote repository is required for piecewise linear encoding of predictor variables. Download it [here](https://github.com/Yura52/rtdl).

Feel free to reach out if you have any questions or need assistance. Happy coding!


## License

Our code is licensed under a GPL version 3 license (see the license file for detail).


## Citation

Please view our publication on JAMIA:

If you find our project useful, please consider citing our work:
```
@article{
}
```
