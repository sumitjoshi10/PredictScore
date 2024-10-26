import os
import sys
from src.exception import CustomeException
from src.logger import logging
from dotenv import load_dotenv
import pymysql
import pandas as pd
import pickle


load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb = pymysql.connect(
            host= host,
            user=user,
            password=password,
            db=db
        )
        logging.info(f"Connection Established to {mydb}")
        df = pd.read_sql_query("select * from students",mydb)
        
        print(df.head())
        return df
        
    except Exception as e:
        logging.info("Unable to read the SQL Data")
        raise CustomeException(e,sys)


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        
        
    except Exception as e:
        raise CustomeException(e, sys)