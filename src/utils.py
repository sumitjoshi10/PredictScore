import os
import sys
from src.exception import CustomeException
from src.logger import logging
from dotenv import load_dotenv
import pymysql
import pandas as pd
import dill
import json

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


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
            dill.dump(obj,file_obj)
        
        
    except Exception as e:
        raise CustomeException(e, sys)
    
    
def save_json(file_path,dictionary):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"w") as file_json:
            json.dump(dictionary,file_json,indent=4)
        
        
    except Exception as e:
        raise CustomeException(e, sys)
    

def evauluate_model(X_train,y_train,X_test,y_test,model_params):
    try:
        scores = {}
        
        for model_name, mp in model_params.items():
            logging.info(f"Starting the Hyper parameter Tunnig for {model_name}")
            clf = GridSearchCV(mp["model"],mp["params"], cv = 5, return_train_score= False)
            clf.fit(X_train,y_train)
            
             # Store results
            best_params = clf.best_params_
            best_score = clf.best_score_
            
            # Calculate R² score on the test set
            best_model = clf.best_estimator_
            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
          
            avg_score = best_score*0.6+r2*0.4
            # scores.append({
            #     "model" : model_name,
            #     "best_score" : best_score,
            #     "best_params": best_params,
            #     "test_r2_score": r2,
            #     "avg_best_score": avg_score
            # })
            scores[model_name]={
                "best_score" : best_score,
                "best_params": best_params,
                "test_r2_score": r2,
                "avg_best_score": avg_score
            }
            logging.info(f"Completed Hyper parameter Tunnig for {model_name} having scores as {best_score} and test R2 score as {r2} with the average best score as {avg_score}")
                   
        logging.info("Hyper parameter Tunning Complete")
        return scores
        
    except Exception as e:
        raise CustomeException(e,sys)