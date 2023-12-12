"""
Project: Variable Length Sequence Modeling
Branch:
Author: Azi Bashiri
Last Modified: Nov 2022
Description: This python module defines Logger class. The Logger constructor instantiates a logger object, creates a
                logging directory at "Proj_dir/Output/log_date-time". A [.txt] log file is optional to use.
"""

import os
import datetime

cwd = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))


class Logger:
    def __init__(self, b_log_txt=False, log_name="log_train.txt", log_path=None, log_folder=None):
        self.b_log_txt = b_log_txt  # boolean, open a [.txt] file if True
        self.log_name = log_name  # name of the [.txt] log file
        if log_path is None or not log_path:
            log_path = os.path.join(proj_dir, "Output")
        if log_folder is None or not log_folder:
            log_folder = 'log_' + datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.log_dir = os.path.join(log_path, log_folder)
        self.log_fout = None  # a file object

        # create (log_date-time) folder if doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        # create .txt log file
        if self.b_log_txt:
            self.log_fout = open(os.path.join(self.log_dir, self.log_name), 'a+')

    def log_string(self, out_str="\n"):
        """
        Print out out_str. If a logging [.txt] file is created (i.e., self.log_fout contains a file object) then out_str
        will be written into the file.
        :param out_str: String to log. If no out_str is provided, a "new line" will be written.
        :return: None
        """
        if self.log_fout is not None:
            self.log_fout.write(out_str+'\n')
            self.log_fout.flush()
        print(out_str)
        return None

    def log_fclose(self):
        if self.log_fout is not None:
            self.log_fout.close()

    def __str__(self):
        s = f"Logger: \n"
        s += f"\t log_dir = {self.log_dir} \n"
        if self.b_log_txt is None:
            s += f"\t log_name = None"
        else:
            s += f"\t log_name = {self.log_name}"
        return s
