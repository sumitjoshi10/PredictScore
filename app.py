from src.logger import logging
from src.exception import CustomeException
import sys


if __name__ == "__main__":
    logging.info("The Execution has been started")
    
    try:
        z = 1/0
    except Exception as e:
        logging.info("Zero Division Error")
        raise CustomeException(e, sys)