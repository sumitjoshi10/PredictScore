from src.logger import logging
from src.exception import CustomeException
import sys

from src.components.data_ingestion import DataIngestion


if __name__ == "__main__":
    logging.info("The Execution has been started")
    
    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("Zero Division Error")
        raise CustomeException(e, sys)