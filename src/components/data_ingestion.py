import os
import sys
import pandas as pd

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import load_dataset, DATASET, label_stratified_folds

@dataclass
class DataIngestionConfig:
    """
    A class with static variables to store splitted data

    Attributes:
        raw_path (str): A static variable that holds the raw data path as string.
        stratified_folds_path (str): A static variable that holds the stratified fold data path as string.
    """
    raw_path = os.path.join("artifacts","data.csv")
    stratified_folds_path = os.path.join("artifacts","stratified_folds.csv")

class DataIngestion:
    """
    A class with an attribute that is set to a DataIngestionConfig.    
    """

    def __init__(self):
        """
        Initializes a new DataIngestionConfig instance.
        """
        self.ingestion_config = DataIngestionConfig()

    def init_data_ingestion(self, dataset_name):
        """ Returns raw and stratified fold paths after data ingestion

        Expects:
            dataset_name: a str that represent the dataset name.
        Modifies:
            Nothing.
        Returns:
            raw_path: a str that represents the raw data path
            stratified_folds_path: a str that represents the stratified fold data path
        """
        logging.info("Entered data ingestion")
        try:
            df, dataset_info = load_dataset(dataset_name)
            logging.info("Read the dataset as dataframe")
            df = df.sample(frac = 1)    

            os.makedirs(os.path.dirname(self.ingestion_config.raw_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_path, index=False, header=True)

            df = label_stratified_folds(df, 10 ,dataset_name)
            logging.info("Stratified folds labeling complete")

            df.to_csv(self.ingestion_config.stratified_folds_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.raw_path,
                self.ingestion_config.stratified_folds_path
            )
        except Exception as e:
            raise CustomException(e, sys)
