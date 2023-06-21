import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from dataclasses import dataclass
from tqdm import tqdm
from src.utils import *
from src.exception import logging

@dataclass
class KNearestNeighborConfig:
    """
    A class with variables for this k-nearest-neighnbor algorithm 
    """
    X_trn = None
    y_trn = None
    X_tst = None
    y_tst = None
    params = None

class KNearestNeighbor:
    """
    A class with an attribute that sets the KNearestNeighborConfig
    """

    def __init__(self):
        self.knn_config = KNearestNeighborConfig()

    def set_params(self, params):
        """ Set params """
        self.knn_config.params = params

    def fit(self, X_trn, y_trn):
        """ Returns training predictions

        Expects: 
            X_trn: a 2-dimensional array that represents training data.
            y_trn: a 2-dimensionsl array that represents testing data.
        Modifies:
            It modifies X_trn and X_tst from knn_config.
        Returns:
            y_preds: a 1D-array with the training predictions.
        """
        self.knn_config.X_trn = X_trn
        self.knn_config.y_trn = y_trn
        k = self.knn_config.params["neighbor"]

        y_preds = []
        for i in range(len(X_trn)):
            dist = []
            row1 = X_trn[i,:]
            for j in range(len(X_trn)):
                row2 = X_trn[j,:]
                dist.append(self.euclidieanDistance(row1, row2))
                
            neighbors = list(zip(dist,y_trn))
            neighbors.sort(key=lambda tup:tup[0])
            
            y_preds.append(self.getPrediction(neighbors[0:k]))
            
        return np.array(y_preds)

    def predict(self, X_tst):
        """ Returns testing predictions

        Expects: 
            X_tst: a 2-dimensional array that represents testing data.
        Modifies:
            Nothing.
        Returns:
            y_preds: a 1D-array with the testing predictions.
        """

        X_trn = self.knn_config.X_trn
        y_trn = self.knn_config.y_trn
        k = self.knn_config.params["neighbor"]

        y_preds = []
        for i in range(len(X_tst)):
            dist = []
            row1 = X_tst[i,:]
            for j in range(len(X_trn)):
                row2 = X_trn[j,:]
                dist.append(self.euclidieanDistance(row1, row2))
                
            neighbors = list(zip(dist,y_trn))
            neighbors.sort(key=lambda tup:tup[0])
            
            y_preds.append(self.getPrediction(neighbors[0:k]))
            
        return np.array(y_preds)

    def standard_deviation(self,accuracy):
        """ Returns the standard deviation """
        std = -999999
        std = np.std(accuracy)
        return std

    def euclidieanDistance (self,row1, row2):
        """ Returns the euclidiean distance between 2 1D-arrays """
        dist = 0.0
        dist = np.sum((row1 - row2)**2)
        return np.sqrt(dist)

    def getPrediction(self, neighbors):
        """ Returns one predicition using the majority vote mechanisim.

        Expects: 
            neighbors: a integer that represents the number of neighbors.
        Modifies:
            Nothing.
        Returns:
            y_preds: aa integer that represents the prediction.
        """

        dict_pred = {}
        
        for neigbor in neighbors:
            # get the lable from the second column
            label = neigbor[1]
            if not label in dict_pred:
                dict_pred[label] = 1
            else:
                dict_pred[label] = dict_pred[label] + 1

        sorted_candidates = sorted(dict_pred.items(), key=lambda tuple: tuple[1])
        
        return sorted_candidates[-1][0]
    
    def grid_search_cv(self, dataset_obj, params, cv, mode=MODE.BREIF):
        """ Returns the best parameters combination with the highest accuracy

        Expects:
            dataset_obj: an object that contains: the dataframe, unique clases, number
                of classes, and dataframe information.
            params: an object with hyperparameters.
            cv: an integer with the number of cross-validation.
            mode: a str that represent the mode: how this method behave.
        Modifies:
            Nothing.
        Returns:
            best_params: an object with the best hyper-parameter settings.
        """
        best_params = {}
        best_accuracy = -99
        try:
            for neighbor_candidate in params["neighbors"]:
                candidate_params = {
                    "neighbor": neighbor_candidate
                }
                self.set_params(candidate_params)

                acc = self.init_cross_validation(dataset_obj, cv, mode=mode)
            
                if acc > best_accuracy:
                    best_accuracy  = acc
                    best_params = candidate_params

            return best_params

        except Exception as e:
            raise CustomException(e, sys)

    def init_cross_validation(self, dataset_obj, cv, mode=MODE.TRAIN):
        """ Returns the average accuracy of the stratified cross validations
        
        Expects:
            dataset_obj: an object that contains: the dataframe, unique clases, number
                of classes, and dataframe information.
            cv: an integer with the number of cross-validation.
            mode: a str that represent the mode: how this method behave.
        Modifies:
            Nothing.
        Returns: the average accuracy of the stratified cross validations
        """

        k_accuracies = 0   
        
        try:
            df = dataset_obj["df"]
            classes = dataset_obj["classes"]
            num_classes = dataset_obj["num_classes"]
            dataset_info = dataset_obj["dataset_info"]
            
            # do cross validation
            for k in (range(int(cv))):
                
                (trn, tst) = train_test_split(df, k+1)
                (X_trn, y_trn, y_trn_trues) = attr_label_split(trn, num_classes)
                (X_tst, y_tst, y_tst_trues) = attr_label_split(tst, num_classes)
                
                if mode == MODE.BREIF:
                    X_trn = X_trn[0:35,:]
                    y_trn_trues = y_trn_trues[0:35]
                    X_tst = X_tst[0:35,:]
                
                # traning
                if mode == MODE.TRAIN or mode == MODE.BREIF :
                    y_preds = self.fit(X_trn, y_trn_trues) 
                
                else: # mode == MODE.TEST:
                    y_preds = self.predict(X_tst)

                ## get the metrics
                confusion_mat = None
                if mode == MODE.TRAIN or mode == MODE.BREIF:
                    confusion_mat = create_confusion_matrix(y_trn_trues, y_preds, classes)
                
                else: # mode == MODE.TEST:
                    confusion_mat = create_confusion_matrix(y_tst_trues, y_preds, classes)
                    
                (fold_acc, __, __, __) = get_multiclass_metrics(confusion_mat)
                
                k_accuracies += fold_acc
                
            return (k_accuracies/cv)
            
        except Exception as e:
            raise CustomException(e, sys)

