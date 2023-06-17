import numpy as numpy
import pandas as pd
import os
import sys
import json

from src.exception import CustomException
from src.logger import logging
from src.utils import trasform_data_set
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    """
    A class with a static variable to store the .pkl file

    Attributes:
        preprocessor_obj_file_path: a string that holds the path of the proprocessor.pkl file
    """

    # not used for now
    preprocessor_obj_file_path = os.path.join("artifact","preprocessor.pkl")

class DataTransformation:
    """
    A class with an attribute that is set to DataTransformationConfig
    """
    def __init__(self):
        """
        Initializes a new DataTransformationConfigs instance.
        """
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer(self):
        """ Returns a data transformer for a given dataset
        
        Since the attributes for a datase are handle in DIC_ATTRIBUTES.      
        (check src.utils.DIC_ATTRIBUTES), storing the preprocesssor.pkl
        file it is not necessary for now.
        """
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys) 

    def initiate_data_transformation(self, stratified_folds_path, dataset_info_json_path):
        """ Returns a dataframe with transformed attributes from a specific path
        
        Expects: 
            stratified_folds_path: a string with the path of the dataframe.
        Modifies: 
            Nothing.
        Returns:
            df: a dataframe with transformed attributes. If categorical attributes are
                present, the shape of the df will be greater than the original dataframe.
            classes: an array with unique classes of the dataframe
            num_classes: a number with the number of unique classes of the df.
        """
        try:
            stratfied_folds = pd.read_csv(stratified_folds_path)
            logging.info("Read stratified folds data completed")

            # reading from dataset_info_json_path
            with open(dataset_info_json_path,"r") as f:
                dataset_info = json.load(f)
            
            (df, classes, num_classes) = trasform_data_set(stratfied_folds, dataset_info)
            logging.info("Data transformation completed")
            
            return (df, classes, num_classes)

        except Exception as e:
            raise CustomException(e, sys)

from src.utils import DATASET
from src.components.data_ingestion import DataIngestion
if __name__=="__main__":
    try:
        data_ingestion = DataIngestion()
        (_, 
        stratified_fold_path, 
        dataset_info_json_path) = data_ingestion.init_data_ingestion(DATASET.LOAN)

        data_transformation = DataTransformation()
        (df, classes, num_classes) = data_transformation.initiate_data_transformation(stratified_fold_path, dataset_info_json_path)
        
    except Exception as e:
        raise CustomException(e, sys)
    