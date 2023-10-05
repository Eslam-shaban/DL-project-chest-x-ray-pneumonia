import os
import urllib.request as request
import zipfile
from pathlib import Path
from lung_cancer import logger
from lung_cancer.utils.common import get_size
from lung_cancer.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  


    def download_file_kaggle(self):
        if not os.path.exists(self.config.local_data_file_kaggle):
            os.system(f'kaggle datasets download -d {self.config.command_api} -p {self.config.local_data_file_kaggle}')
            logger.info(f"Dataset downloaded to {self.config.local_data_file_kaggle}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file_kaggle))}")

    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            
            
    def extract_zip_file_kaggle(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        # Find the .zip file in the directory
        zip_file = None
        for file in os.listdir(self.config.root_dir):
            if file.endswith(".zip"):
                zip_file = os.path.join(self.config.root_dir, file)
                break

        if zip_file is None:
            raise FileNotFoundError("No .zip file found in the directory")

        # Extract the .zip file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
