"""
Project: Variable Length Sequence Modeling
Branch: --
Author: Azi Bashiri
Last Modified: July 2022
Description: A simple script that shows how to use Logger class from my_logger.py
"""

import Code.VL011_globals as glb

if __name__ == '__main__':
    # create a logger object in the form of a global variable. A global variable can be used outside of main script,
    # such as mdl_xxx.py scripts
    glb.init_logging(b_log_txt=True, log_name="my_log.txt")
    # log a string
    glb.logger.log_string(f"We are testing!")
    glb.logger.log_string(glb.logger.__str__())
    # close the log file
    glb.logger.log_fclose()
