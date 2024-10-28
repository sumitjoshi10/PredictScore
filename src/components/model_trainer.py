import os
import sys
from dataclasses import dataclass

from src.exception import CustomeException
from src.logger import logging

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.utils import save_object,evauluate_model,save_json

import ast

@dataclass
class ModelTrainerConfig:
    best_model_file_path = os.path.join("artifacts","best_models.json")
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    
    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("Split Training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            # Model Training along with Hyper parameter tuning
            # Creating a dictionary
            model_params = {
                "Random_Forest":{
                    "model":RandomForestRegressor(),
                    "params":{
                        "n_estimators": [100, 200, 500, 1000],
                        # "criterion": ["squared_error", "absolute_error", "poisson"],
                        # "max_features": ["auto", "sqrt", "log2", 0.5, 0.75]
                    }
                },
                "Decision_Tree":{
                    "model":DecisionTreeRegressor(),
                    "params":{
                        "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                        # "splitter": ["best", "random"],
                        # "max_features": [None, "auto", "sqrt", "log2", 0.5, 0.75]
                    }
                },
                "Gradient_Boosting":{
                    "model":GradientBoostingRegressor(),
                    "params":{
                        # "loss": ["squared_error", "absolute_error", "huber", "quantile"],
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "n_estimators": [100, 200, 500, 1000],
                        "subsample": [1.0, 0.8, 0.6, 0.4],
                        # "criterion": ["friedman_mse", "squared_error"],
                        # "max_features": [None, "auto", "sqrt", "log2", 0.5, 0.75]
                    }
                },
                "Linear_Regression":{
                    "model":LinearRegression(),
                    "params":{
                    }
                },
                "XG_Boosting":{
                    "model":XGBRegressor(),
                    "params":{
                        "n_estimators": [100, 200, 500, 1000], 
                        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                    }
                },
                "KNN":{
                    "model":KNeighborsRegressor(),
                    "params":{
                        "n_neighbors":[3,5,7,9,11,13],
                        # "weights":["uniform",'distance'],
                        # "p":[1,2]
                    }
                },
                "Cat_Boost":{
                    "model":CatBoostRegressor(),
                    "params":{
                        "iterations": [1000, 2000], 
                        # "iterations": [100, 500, 1000, 2000], 
                        "learning_rate": [0.01, 0.05, 0.1, 0.2], 
                        "depth": [4, 6, 8, 10, 12]
                    }
                },
                "AdaBoost":{
                    "model":AdaBoostRegressor(),
                    "params":{
                        "n_estimators": [50, 100, 200, 500],
                        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                        # "loss": ["linear", "square", "exponential"],
                    
                    }
                }
            }
            
            scores = evauluate_model(X_train,y_train,X_test,y_test,model_params)
            save_json(
                file_path=self.model_trainer_config.best_model_file_path,
                dictionary=scores
            )
            
            # Will get the best model name from the below.
            best_model_name = max(scores, key=lambda k: scores[k]['avg_best_score'])
            best_model_details = scores[best_model_name]
            
            # best_score_model = scores_df[scores_df['avg_best_score'] == scores_df['avg_best_score'].max()].to_dict(orient='records')[0]
            
            best_model = model_params[best_model_name]["model"]
            best_param = best_model_details["best_params"]
            
            best_model.set_params(**best_param)
            
            if best_model_details["test_r2_score"]<0.6:
                raise CustomeException("No best model found")
            
            logging.info("Best model found for both training and test data set")
            logging.info(f"{best_model} Model has been selected for model Training")
            
               
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            best_model.fit(X_train,y_train)
            prediction = best_model.predict(X_test)
            r2 = r2_score(y_test,prediction)
            
            logging.info(f"[{best_model}] Model with parameter [{best_model_details["best_params"]}] has been selected for model Training. It got [{r2}] score.")
            
        except Exception as e:
            raise CustomeException(e,sys)
