import os
import sys
from src.exception import CustomeException
from src.logger import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.utils import save_object


@dataclass
class DataTransformerConfig:
    prepocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformerConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        
        try:
            logging.info("Numerical Columns Standard Scaling Started")
            logging.info("Categorical Columns Encoding Started")
            # numerical and categorical columns
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            logging.info(f"Numerical Columns: {numerical_columns}")
            logging.info(f"Categorical Columns: {categorical_columns}")
            
            # Numerical and Categorical Pipelines
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler())
                ]
            )
            
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Numerical Columns Standard Scaling Completed")
            logging.info("Categorical Columns Encoding Completed")
            
            prepocessor = ColumnTransformer([
                ("numerical_pipeline",numerical_pipeline,numerical_columns),
                ("categorical_pipeline",categorical_pipeline,categorical_columns)
            ])
            
            return prepocessor
            
        
        except Exception as e:
            logging.info("Error occured during data transformation")
            raise CustomeException(e,sys)
    
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Traing and Test data set
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read Train and Test data completed")
            
            logging.info("Obtaining the preprocessor object")
            
            # Creating the prepocessor object
            preprocessor_obj = self.get_data_transformer_object()
            
            # Dependent and Independent Feature 
            target_columns = "math_score"
            
            input_feature_train_df = train_df.drop(columns=[target_columns],axis=1)
            target_feature_train_df = train_df[target_columns]
            
            input_feature_test_df = test_df.drop(columns=[target_columns],axis=1)
            target_feature_test_df = test_df[target_columns]
        
            logging.info("Applying preprocessing in train and test data set")
        
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            # Concatinating the Transformed Array and Target Column
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing objects")
            
            save_object(
                file_path=self.data_transformation_config.prepocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.prepocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomeException(e, sys)