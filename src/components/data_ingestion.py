import os
import sys
from src.exception import CustomeException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split 

from src.utils import read_sql_data

from dataclasses import dataclass

@dataclass
class DataIngestionCongifg:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","raw.csv")
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionCongifg()
        
    def initiate_data_ingestion(self):
        try:
            # Reading from the MySQL Database
            df = read_sql_data()
            logging.info("Reading  Completed for the MySQL Database")
            
            # To Create a artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True) 
            
            # Save the Raw data to the raw_data_path
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            # Train Test Split the data
            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42)
            
            # Saving the Train and Test Data into the path
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Data Ingestion is completed")
            
        except Exception as e:
            logging.info("Error occured during data ingestion")
            raise CustomeException(e,sys)