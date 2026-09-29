import sys


def error_message_details(error, error_details:sys):
    # error_details.exc_info() will return three values as tuple (type, value, traceback)
    _,_,exc_tb=error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f'Error occured in {file_name} file, in {exc_tb.tb_lineno} line and error message - {str(error)}'
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_details):
        super().__init__(error)
        self.error_message = error_message_details(error, error_details)
    def __str__(self):
        return self.error_message
    