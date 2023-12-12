"""
In order to use a same logger object in different python scripts including any mdl_xxx.py file, it has to be a global
variable. A global variable that is defined in VL011_globals.py can be instantiated in main.py, and be used anywhere
"""
from Code.My_logger.my_logger import Logger


def init_logging(b_log_txt=False, log_name="log_train.txt", log_path=None, log_folder=None):
    # reference on using global variables b/w files in python:
    # https://stackoverflow.com/questions/13034496/using-global-variables-between-files
    global logger
    logger = Logger(b_log_txt, log_name, log_path=log_path, log_folder=log_folder)
    return
