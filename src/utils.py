import numpy as np
import pandas as pd
from sklearn import datasets 
import numpy as np 
import matplotlib. pyplot as plt
from src.logger import logging
import os

K_FOLD = "k_fold"
TARGET_COLUMN = "target"
ATTR_TYPE = "attr_type"
ATTRIBUTES = "attributes"
NAME = "dataset_name"
DATASETS_PATH = os.path.join(os.getcwd(),"notebooks","datasets")

# an array that holds raw target names with respect to all datasets
RAW_TARGET_NAMES = [
    "# class", "class", "Class", "Loan_Status", "Diagnosis", "Survived", 64]

class DATASET:
    """
    A class with static variables to handle dataset names
    """
    IRIS = "iris"
    WINE = "wine"
    VOTE = "vote"
    CANCER = "cancer"
    CONTRACEPTIVE = "contraceptive"
    LOAN = "loan"
    PARKINSONS = "parkinson"
    TITANIC = "titanic"
    DIGITS = "digits"

class METRIC:
    ACCURACY = "accuracy"
    ERROR_RATE = "error rate"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1 score"
    
class ATTRIBUTE_TYPE :
    """
    A class with static variables to handle different types of attributes
    """
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    NUMERICAL_AND_CATEGORICAL = "numerical_and_categorical"

# DIC_ATTRIBUTES = {
DIC_DATASETS_INFO = {
    DATASET.IRIS:{
        ATTR_TYPE : ATTRIBUTE_TYPE.NUMERICAL,
        NAME : DATASET.IRIS
        
    },
    DATASET.WINE:{
        ATTR_TYPE : ATTRIBUTE_TYPE.NUMERICAL,
        NAME : DATASET.WINE
        
    },
    DATASET.VOTE:{
        ATTR_TYPE : ATTRIBUTE_TYPE.CATEGORICAL,
        NAME : DATASET.VOTE
    },
    DATASET.CANCER: {
        ATTR_TYPE: ATTRIBUTE_TYPE.NUMERICAL,
        NAME : DATASET.CANCER
    },
    DATASET.CONTRACEPTIVE:{
        ATTR_TYPE : ATTRIBUTE_TYPE.NUMERICAL_AND_CATEGORICAL,
        NAME : DATASET.CONTRACEPTIVE,
        ATTRIBUTES:{
            'wife_age': {
                "type": ATTRIBUTE_TYPE.NUMERICAL, "delete" :False },
            "wife_edu": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False },
            "hsbnd_edu": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False },
            "num_chldr": {
                "type": ATTRIBUTE_TYPE.NUMERICAL, "delete" :False },
            "wife_rel": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False },
            "wife_not_work": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False },
            "hsbnd_occup": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False },
            "st_liv": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False },
            "media_exp": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False },
            "target": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False}
        }
    },
    DATASET.LOAN:{ 
        ATTR_TYPE : ATTRIBUTE_TYPE.NUMERICAL_AND_CATEGORICAL,
        NAME : DATASET.LOAN,
        ATTRIBUTES:{
            'Loan_ID': {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :True },
            "Gender": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False },
            "Married": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False },
            "Dependents": {
                "type": ATTRIBUTE_TYPE.NUMERICAL, "delete" :False },
            "Education": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False },
            "Self_Employed": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False },
            "ApplicantIncome": {
                "type": ATTRIBUTE_TYPE.NUMERICAL, "delete" :False },
            "CoapplicantIncome": {
                "type": ATTRIBUTE_TYPE.NUMERICAL, "delete" :False },
            "LoanAmount": {
                "type": ATTRIBUTE_TYPE.NUMERICAL, "delete" :False },
            "Loan_Amount_Term": {
                "type": ATTRIBUTE_TYPE.NUMERICAL, "delete" :False },
            "Credit_History": {
                "type": ATTRIBUTE_TYPE.NUMERICAL, "delete" :False },
            "Property_Area": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False },
            "target": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False }
        }
    },
    DATASET.PARKINSONS:{ 
        ATTR_TYPE : ATTRIBUTE_TYPE.NUMERICAL,
        NAME : DATASET.PARKINSONS
    },
    DATASET.TITANIC:{ 
        ATTR_TYPE : ATTRIBUTE_TYPE.NUMERICAL_AND_CATEGORICAL,
        NAME : DATASET.TITANIC,
        ATTRIBUTES:{
            'Pclass': {
                "type": ATTRIBUTE_TYPE.NUMERICAL, "delete" :False },
            "Name": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :True },
            "Sex": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False },
            "Age": {
                "type": ATTRIBUTE_TYPE.NUMERICAL, "delete" :False },
            "Siblings/Spouses Aboard": {
                "type": ATTRIBUTE_TYPE.NUMERICAL, "delete" :False },
            "Parents/Children Aboard": {
                "type": ATTRIBUTE_TYPE.NUMERICAL, "delete" :False },
            "Fare": {
                "type": ATTRIBUTE_TYPE.NUMERICAL, "delete" :False },
            "target": {
                "type": ATTRIBUTE_TYPE.CATEGORICAL, "delete" :False }
        }
    },
    DATASET.DIGITS: {
        ATTR_TYPE: ATTRIBUTE_TYPE.NUMERICAL,
        NAME : DATASET.DIGITS
    }
}

def load_dataset(dataset_name):
    """ Loads a given dataset by name.
    
    Expects: 
        dataset_name: a string with the dataset name.
    Modifies: 
        It renames and places the label at the end of the dataframe.
    Return: 
        df: a datset as a dataframe.
        dataset_info : a dictionary with dataset information.
    """

    df = pd.DataFrame()
    dataset_info = None
    
    match dataset_name :
        case DATASET.IRIS:
            df = pd.read_csv(
                os.path.join(DATASETS_PATH,"iris.csv"),
                names=[
                    "sepal_length","sepal_width",
                    "petal_length","petal_withd","class"])
            dataset_info = DIC_DATASETS_INFO[DATASET.IRIS]
            
        case DATASET.WINE:
            df = pd.read_csv(os.path.join(DATASETS_PATH,"wine.csv"),sep="	")
            dataset_info = DIC_DATASETS_INFO[DATASET.WINE]
            
        case DATASET.VOTE:
            df = pd.read_csv(os.path.join(DATASETS_PATH,"vote.csv"))
            dataset_info = DIC_DATASETS_INFO[DATASET.VOTE]
        
        case DATASET.CANCER:
            df = pd.read_csv(os.path.join(DATASETS_PATH,"cancer.csv"),sep="	")
            dataset_info = DIC_DATASETS_INFO[DATASET.CANCER]
        
        case DATASET.CONTRACEPTIVE:
            df = pd.read_csv(
                os.path.join(DATASETS_PATH,"cmc.data"),
                sep=",", 
                names=[
                    'wife_age', 'wife_edu', 'hsbnd_edu',
                    'num_chldr','wife_rel','wife_not_work',
                    'hsbnd_occup','st_liv','media_exp','target'])
            dataset_info = DIC_DATASETS_INFO[DATASET.CONTRACEPTIVE]

        case DATASET.LOAN:
            df = pd.read_csv(os.path.join(DATASETS_PATH,"loan.csv"))
            dataset_info = DIC_DATASETS_INFO[DATASET.LOAN]
        
        case DATASET.PARKINSONS:
            df = pd.read_csv(os.path.join(DATASETS_PATH,"parkinsons.csv"))
            dataset_info = DIC_DATASETS_INFO[DATASET.PARKINSONS]
        
        case DATASET.TITANIC:
            df = pd.read_csv(os.path.join(DATASETS_PATH,"titanic.csv"))
            dataset_info = DIC_DATASETS_INFO[DATASET.TITANIC]

        case DATASET.DIGITS:
            df = load_digits_data_set() # loads from internet
            dataset_info = DIC_DATASETS_INFO[DATASET.DIGITS]
        
        case _:
            logging.info("something went wrong")

    # standarization of class name and relocation as last column
    # of the dataframe
    df = set_true_class_as_last_column(df, RAW_TARGET_NAMES)
    
    return (df, dataset_info)

def set_true_class_as_last_column(df, true_class_names):
    """ Normalize the name of the label and place it the end of the df.
    
    Expects: 
        df: a dataframe where to look for the class name
        true_class_names: an array with all possible class names found in 
            each datasets
    Modifies: 
        It modifies the dataframe by adding a new column called target whose 
        values are equal to the class name, then drop the class name.    
    Return:
        df: a dataframe with the a new column called target.
    """
    for true_class_name in true_class_names:
        # if this true_class_name exit in df, 
        # set it as a last column
        if df.iloc[:,df.columns == true_class_name ].size != 0:
            df[TARGET_COLUMN] = df.iloc[:,df.columns == true_class_name]
            return df.drop(columns=[true_class_name])
    return df

# def normalize_target(df, dataset_name):
def map_target_values_as_intergers(df, dataset_name):
    """ Returns a dataframe with target values as integers.
        
        Expects: 
            df: a dataframe that holds the string target values.
            dataset_name: a string with the datset name.
        Modifies: 
            It modifies the target column by replacing the string values 
            into integer values.
        Returns: 
            df: a dataframe with target values as integers.
    """
    match dataset_name :
        case DATASET.IRIS:
            target_map = {
                "Iris-setosa": 0 ,
                "Iris-versicolor" : 1, 
                "Iris-virginica" : 2 
            }
            df[TARGET_COLUMN] = df[TARGET_COLUMN].map(target_map)
            
        case DATASET.LOAN:
            target_map = { "Y": 1 , "N" : 0}
            dependants_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
            df[TARGET_COLUMN] = df[TARGET_COLUMN].map(target_map)
            df["Dependents"] = df["Dependents"].map(dependants_map)

        case DATASET.DIGITS:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df

def label_stratified_folds(df, k, dataset_name):
    """ Returns a dataframe with a stratified k-fold column.

    Expects: 
        df: a dataframe.
        k: a integer th                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     at represents the number of k-folds.
        dataset_name: a string with the datset name.
    Modifies:
        It appends the k-fold column and each row is labeled according the 
        stratified sampling, which ensures the same proportions of target 
        values as in  the original dataframe. 
    Returns: 
        df: a dataframe with the k-fold appended at the end.
    """
    # normalize the target column into class numbers
    df = map_target_values_as_intergers(df, dataset_name)
    # get the proportion of each class
    classes_proportion = np.bincount(df[TARGET_COLUMN]) 
    
    for y_class in range(len(classes_proportion)):

        # split ids based on the class of interest 
        ids = df[df[TARGET_COLUMN] == y_class].index

        num_partition = round(len(ids)/k)
        low_range = 0
        upper_range = num_partition
        
        # traverse through each k value and split the ids array based on
        # the dynamic ranges, so we can label each instance sequencially  
        # to one particular k-fold value 
        for j in range(k):

            k_fold_label = j + 1 
    
            # traverse each dynamic range: lower and upper bound
            for i in range(low_range, upper_range):
                df.at[ids[i],K_FOLD] = k_fold_label

            # updating ranges
            low_range = upper_range 

            # Last k-fold: when the second to the last k-fold (k - 1) value,
            # we set upper range to the lenght of the ids arrays, so we get
            # all the reminders ids for the last k-fold
            if (j+1 == k-1):
                upper_range = len(ids)
            else:
                upper_range += num_partition
    return df

# def retrieve_trn_tst_dataset(df, k):
def train_test_split(df, k):
    """ Returns train and test sets as dataframes based on k-fold value.

    Expects: 
        df: a dataframe with k-fold column populated.
        k: an integer that represent the k-fold of interest.
    Modifies: 
        Nothing.
    Returns:
        trn: a dataframe which k-fold values is diffent than k value.
        tst: a dataframe which k-fold values is equal to k value.
    """
    trn = df[df[K_FOLD] != k]
    tst = df[df[K_FOLD] == k]
    return (trn, tst)

def get_metric(metric, TP, FN, FP, TN):
    """ Returns the metric value of interest.

    Expects:
        metric: a string that specify the metric of interest.
        TP: a integer that specifies the True Positives.
        FN: a integer that specifies the False Negatives.
        FP: a integer that specifies the False Positives.
        TN: a integer that specifies the True Negatives.
    Modifies: 
        Nothing.
    Returns: 
        value: a float that represents the metric.
    """
    
    value = 0.0
    n = TP + FN + FP + TN
    
    match metric:
        case METRIC.ACCURACY:
            value = (TP + TN) / n
            
        case METRIC.ERROR_RATE:
            value = (FP + FN) / n
            
        case METRIC.PRECISION:
            if TP + FP != 0:
                value = TP / (TP + FP )
            
        case METRIC.RECALL:
            if TP + FN != 0:
                value = TP / (TP + FN) 
    
        case METRIC.F1_SCORE:
            
            prec = get_metric(METRIC.PRECISION, TP, FN, FP, TN)
            recall = get_metric(METRIC.RECALL, TP, FN, FP, TN)
            
            if prec + recall != 0: # cero is default
                value = (2 * prec * recall) / (prec + recall)
            
    return value

def get_multiclass_metrics(confusion_mat):
    """ Returns metrics of a multiclass.

    Expects:
        confusion_mat: a matrix of integers.
    Modifies:
        Nothing.
    Returns: 
        accuracy: a float the represents the accuracy.
        precision: a float the represents the precision.
        recall: a float the represents the recall.
        f_1_score: a float the represents the f_1_score.

    """
    confusion_mat = np.array(confusion_mat)
    acc = pre = rec = f1 = 0
    len_mat = len(confusion_mat)
    
    for i in range(len_mat):
        TP = FN = FP = TN = 0
        
        TP = confusion_mat[i,i]
        FP = sum(confusion_mat[:,i]) - TP
        FN = sum(confusion_mat[i,:]) - TP
        TN = sum(sum(confusion_mat)) - TP - FP - FN
        
        acc += get_metric(METRIC.ACCURACY, TP, FN, FP, TN)
        pre += get_metric(METRIC.PRECISION, TP, FN, FP, TN)
        rec += get_metric(METRIC.RECALL, TP, FN, FP, TN)
        f1 += get_metric(METRIC.F1_SCORE, TP, FN, FP, TN)

    return (acc/len_mat , pre/len_mat, rec/len_mat, f1/len_mat)


def create_confusion_matrix(y_true, y_pred, classes):
    """ Returns a confusion matrix.
    
    Expects:
        y_true: an array of true labels.
        y_pred: an array of predicted labels.
        classes: an array of unique labels.
    Modifies:
        Nothing.
    Returns:
        matrix: a matrix that represent the confusion matrix.
    """
    
    # when classes are values other than 0 (Erg.[1,2,3,4]) we substract 1
    # from them so they match with the confusion matrix indexes
    add_flag = 1
    
    # for classes containing 0 as a class/label, we set the flag to zero
    if 0 in classes:
        add_flag = 0
    
    matrix = [ [0 for j in range(len(classes))] for i in range(len(classes))]

    for i in range(len(y_pred)):
        matrix[y_true[i] - add_flag ][y_pred[i] - add_flag] += 1

    return matrix


def plot_confusion_matrix(TP,FN,FP,TN):
    """ Returns confusion matrix plot.

    Expects:
        TP: a integer that specifies the True Positives.
        FN: a integer that specifies the False Negatives.
        FP: a integer that specifies the False Positives.
        TN: a integer that specifies the True Negatives.
    Modifies: 
        Nothing.
    Returns: a plot with the confusion matrix.
    """
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix = np.array([[TP,FN,FN],[FP,TN,TN]]), 
        display_labels = [True, False])
        
    cm_display.plot()
    plt.show()

# def preprocess_data_set(df, dataset_info):
def trasform_data_set(df, dataset_info):
    """ Returns a dataframe with transformed attributes
    
    Expects: 
        df: a dataframe where the attribute are read from.
        dataset_info: an object that holds the attribute types (numerical or categorical).
    Modifies: 
        It normalize and perform one hot enconding technique to 
        numerical and categorical attributes respectively.
    Returns:
        df: a dataframe with transformed attributes. If categorical attributes are
            present, the shape of the df will be greater than the original dataframe.
        classes: an array with unique classes of the dataframe.
        num_classes: a number with the number of unique classes of the df.
    """
    
    num_classes = len(np.bincount(df.iloc[:,-2]))
    classes = np.unique(df[TARGET_COLUMN])
    
    target = df.iloc[:,df.columns == TARGET_COLUMN]
    target_one_hot_enc = pd.get_dummies(target,
                                        columns=[TARGET_COLUMN],
                                        dtype=int)
    k_fold = df.iloc[:,df.columns == K_FOLD]
    

    match dataset_info[ATTR_TYPE]:
        case ATTRIBUTE_TYPE.NUMERICAL:
            
            num_attr = df.iloc[:, 0:-2]
            df_norm = normalize(num_attr)
            df = pd.concat([df_norm, target_one_hot_enc, target, k_fold],axis=1)

            if dataset_info[NAME] == DATASET.WINE: ## global variable
                num_classes = num_classes  -1 # from [0,1,2,3] to [1,2,3] class 0 removed
            
        case ATTRIBUTE_TYPE.CATEGORICAL:
            
            columns = df.columns[0:-2].values
            attr_one_hot_enc = pd.get_dummies(df.iloc[:,0:-2],columns=columns, dtype=int)
            df = pd.concat([attr_one_hot_enc, target_one_hot_enc, target, k_fold],axis=1)
            
        case ATTRIBUTE_TYPE.NUMERICAL_AND_CATEGORICAL:
            
            df_preprocessed_data = pd.DataFrame()
            
            # do not traverse tha last who columns(target and k-fold)
            for attr_name in dataset_info[ATTRIBUTES]:
                if ( not dataset_info[ATTRIBUTES][attr_name]["delete"] ):
                    
                    # for numerical attributes base on the attribute data type
                    if  ATTRIBUTE_TYPE.NUMERICAL == dataset_info[ATTRIBUTES][attr_name]["type"]:
                        df_norm_attr = normalize(df.iloc[:, df.columns == attr_name])
                        df_preprocessed_data = pd.concat([df_preprocessed_data, df_norm_attr], axis=1)

                    # for categorical attributes base on the attribute data type
                    else:
                        df_one_hot_encoded = pd.get_dummies(df.iloc[:, df.columns == attr_name],
                                                            columns=[attr_name],
                                                            dtype=int)
                        df_preprocessed_data = pd.concat([df_preprocessed_data, df_one_hot_encoded],axis =1)
            df = pd.concat([df_preprocessed_data, target, k_fold], axis =1)
            

    return (df, classes, num_classes)


# def preprocess_trn_data_set(df, dataset_info, num_class):
def attr_label_trn_split(df, num_class):
    """ Returns a X train and y train dataframes.

    Expects: 
        df: a dataframe that represents the train set.
        num_class: an integer with number of classes.
    Modifies:
        Nothing.
    Returns: 
        X_trn: a ndarray of arrays with train set attributes.
        y_trn: a ndarray of arrays with train set labels.
    """

    X_trn = y_trn = None
    # lower_limit: number of column we do not need to process 
    # from the end th beginnig of the df
    lower_limit = upper_limit = 2 
    # increment the number of column by the number of classes 
    # to get a limit to process attribute only
    lower_limit += num_class 
    # cut the ranges we care for each set
    X_trn = df.iloc[:, 0:-lower_limit].values
    y_trn = df.iloc[:, -lower_limit : -upper_limit].values
    
    return (X_trn, y_trn)

# def preprocess_tst_data_set(df, dataset_info, num_class):
def attr_label_tst_split(df, num_class):
    """ Returns X test, y test, and 1D array labels from dataframes.

    Expects: 
        df: a dataframe that represents the test set
        num_class: an integer with number of classes.
    Modifies:
        Nothing.
    Returns: 
        X_tst: a ndarray of arrays with test set attributes.
        y_tst: a ndarray of arrays with test set labels.
        y_trues: a 1D array of test set labels.
    """
    X_tst = y_tst = y_trues = None
    # lower_limit: number of column we do not need to process 
    # from the end th beginnig of the df
    lower_limit = upper_limit = 2 
    # increment the number of column by the number of classes 
    # to get a limit to process attribute only
    lower_limit += num_class 
    # cut the ranges we care for each set
    X_tst = df.iloc[:, 0:-lower_limit].values
    y_tst = df.iloc[:, -lower_limit : -upper_limit].values

    # return y_trues as well for the confusion matrix
    target = df.iloc[:,df.columns == TARGET_COLUMN].values
    y_trues = target.reshape((target.shape[0]),)
    
    return (X_tst, y_tst, y_trues)

def get_y_preds_based_on_dataset(y_preds, dataset_info):
    """ Returns predictions that matches the unique labels of the dataset.
    
    Expects: 
        y_preds: a 1D array that contains instances predictions.
        dataset_info: an object that holds datas set information.
    Modifies:
        Nothing.
    Returns: 
        y_preds: a 1D array with the predictions that matches the
        label unique values.
    """
    
    match dataset_info[NAME]:
        case DATASET.IRIS:
            y_preds = np.argmax(y_preds,axis=1)
            
        case DATASET.WINE:
            # adding one to match the class value
            y_preds = np.argmax(y_preds,axis=1)+1 
            
        case DATASET.VOTE:
            y_preds = np.argmax(y_preds,axis=1)

        case DATASET.CANCER:
            y_preds = np.argmax(y_preds,axis=1)
        
        case DATASET.CONTRACEPTIVE:
            y_preds = np.argmax(y_preds,axis=1)+1
        
        case DATASET.LOAN:
            y_preds = np.argmax(y_preds,axis=1)
        
        case DATASET.TITANIC:
            y_preds = np.argmax(y_preds,axis=1)
            
        case DATASET.PARKINSONS:
            y_preds = np.argmax(y_preds,axis=1)
            
        case DATASET.DIGITS:
            y_preds = np.argmax(y_preds,axis=1)
            
    return y_preds

def normalize(X):
    """ Returns a dataframe with normalized features.

    Expects:
        X: a dataframe that holds the unscaled attributes
    Modifies:
        Nothing.
    Returns:
        df: a dataframe with normalized features 
    """
    col_names = X.columns
    X = X.values

    # iterate throug all columns but class column
    norm_col = None
    for col in range(X.shape[1]):
        if X[:, col].max()  == 0:
            norm_col = np.zeros((len(X), ), dtype= float)
        else :
            norm_col = (X[:,col] - X[:,col].min())/(X[:,col].max() - X[:,col].min())
        X[:, col] = norm_col

    if X.shape[1] == 1: # if the X has one column
        df = pd.DataFrame(norm_col)
    else: # if the X has more than one column
        df = pd.DataFrame(X)

    df.columns = col_names
    
    return df

