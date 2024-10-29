import os
from pathlib import Path
from src.logger import logging

logging.info("Files and Folder Creation Started")

list_of_files = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_monitoring.py",
    "src/components/model_trainer.py",
    "src/pipelines/__init__.py",
    "src/pipelines/training.py",
    "src/pipelines/prediction.py",
    "src/exception.py",
    "src/logger.py",
    "src/utils.py",
    "templates/index.html",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    ".env"
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir , file_name = os.path.split(file_path)
    
    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for the file {file_name}")
        
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path,"w") as f:
            pass
            logging.info(f"Creating empty file: {file_path}")
            
    else:
        logging.info(f"{file_name} already exist")


logging.info(" Files and Folder Created")