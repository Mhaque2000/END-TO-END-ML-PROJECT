import sys
from src.exception import CustomException
from src.logger import logging
file = []
with open('requirements.txt') as f:
    file = f.readlines()
print(file)

try:
    result = 10/0
except Exception as e:
    logging.info("failed to execute")
    raise CustomException(e, sys)