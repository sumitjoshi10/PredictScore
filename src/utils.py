import os
import sys
from src.exception import CustomeException
from src.logger import logging
from dotenv import load_dotenv
import pymysql
import pandas as pd

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
