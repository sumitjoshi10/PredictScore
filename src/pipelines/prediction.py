import sys
import os
import pandas as pd

from src.exception import CustomeException
from src.logger import logging

from dataclasses import dataclass

from src.components.data_transformation import DataTransformerConfig
from src.components.model_trainer import ModelTrainerConfig

from src.utils import load_object


@dataclass
class PredictionPipelineConfig:
    prepocessor_path_obj = DataTransformerConfig()
    model_path_obj = ModelTrainerConfig()
    prepocessor_path = prepocessor_path_obj.prepocessor_obj_file_path
    model_path = model_path_obj.trained_model_file_path

class PredictPipeline:
    def __init__(self):
        self.prediction_pipeline_config = PredictionPipelineConfig()
        
    def predict(self, features):
        try:
            logging.info("Loading Preprocessor and perfoming the Preprocessing")
            preprocessor = load_object(
                file_path=self.prediction_pipeline_config.prepocessor_path
                )
            data_scaled = preprocessor.transform(features)
            logging.info("Preprossing Completed")
            
            logging.info("Loading the Model and Performing the prediction")
            model = load_object(
                file_path=self.prediction_pipeline_config.model_path
                )
            pred = model.predict(data_scaled)
            logging.info(f"Predinction completed and the math score is {pred}")
            
            return pred
        except Exception as e:
            raise CustomeException(e, sys)
        
class CustomeData:
    def __init__(self,
                gender:str,
                race_ethnicity:str,
                parental_level_of_education:str,
                lunch:str,
                test_preparation_course:str,
                reading_score:float,
                writing_score:float
                 ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score=reading_score
        self.writing_score = writing_score
    
    def get_data_as_data_frame(self):
        try:
            logging.info("Reading the value from the Webpage")
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "reading_score" : [self.reading_score],
                "writing_score": [self.writing_score]
            }
            logging.info("Successfully Read all the value")
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomeException(e, sys)