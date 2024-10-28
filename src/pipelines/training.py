from src.logger import logging
from src.exception import CustomeException
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    logging.info("The Execution has been started")
    
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path=train_data_path,test_path=train_data_path)
        
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr,test_arr)
        
    except Exception as e:
        raise CustomeException(e, sys)
    