import os
import sys

from dataclasses import dataclass
from src.algorithms.knn import KNearestNeighbor
from src.algorithms.neural_network import NeuralNetwotk
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

    def initiate_model_trainer(self, dataset_obj, mode):
        """ Returns an accuracy report of all the model   
        
        Expects:
            dataset_obj: an object that contains: the dataframe, unique clases, number
                of classes, and dataframe information.
            mode: a string that represents the mode: TRAIN of BREIF. The latter is used for 
                development procedures. In BREIF mode, the datatset is shorten to 35 intances.
        Modifies:
            Nothing.
        Returns:
            model_report: an object with the highest accuracies per model.
            model_best_params:  an object with the best parameters per model.
        """

        logging.info("Model trainer initiated")
        try:
            knn = KNearestNeighbor()
            nn = NeuralNetwotk()
            params = {
                "knn":{"neighbors": [i for i in range(1,4,2)]},
                "nn":{
                    "architectures": [
                        [0,2,2,0],
                        [0,4,2,0],
                        [0,8,8,0]],
                    "learning_rates": [0.2, 0.3, 0.5, 0.8],
                    "lambdas": [0.0001,0.001, 0.01, 0.1, 1, 10],
                    "batch_sizes": [32, 64, 128, 256],
                    "num_epochs": 5
                }

            }

            models = {
                "K-Nearest Neighbor" : knn,
                "Neural Network": nn
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

            return (model_report, model_best_params)

        except Exception as e:
            raise CustomException(e, sys)
