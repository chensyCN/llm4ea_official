import logging
import sys


def log_all_printed_info(func):
    def wrapper(*args, **kwargs):
        log_file_name = kwargs.get('log_file_name')
        if log_file_name is None:
            log_file_name = func.__name__ + '.log'
            
        orig_stdout = sys.stdout
        with open(log_file_name, 'w') as log_file:
            sys.stdout = log_file
            result = func(*args, **kwargs)
        sys.stdout = orig_stdout
        
        return result
    return wrapper


