import os
import sys

from dataclasses import dataclass
from src.algorithms.knn import KNearestNeighbor
from src.utils import MODE, evaluate_models, save_to_pkl
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    """
     A class with a static variable to store the .pkl file
    """
    trained_model_file_path = os.path.join("artifacts","models.pkl")

class ModelTrainer:
    """
    A class with an attribute that is set to ModelTrainerConfig
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # def initiate_model_trainer(self, df, classes, num_classes, dataset_info):
    def initiate_model_trainer(self, dataset_obj, mode):

        logging.info("Model trainer initiated")
        try:
            knn = KNearestNeighbor()
            params = {
                "knn":{"neighbors": [i for i in range(1,4,2)]}
            }

            models = {
                "k-Nearest Neighbor" : knn
            }

            (model_report,
            model_best_params) = evaluate_models(
                                            dataset_obj,
                                            models,
                                            params,
                                            cross_validation=False,
                                            mode=mode)

            # get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            # get the best model name from dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logging.info("Model trainer completed")

            save_to_pkl(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            return model_report

        except Exception as e:
            raise CustomException(e, sys)
