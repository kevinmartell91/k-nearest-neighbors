from tqdm import tqdm
from dataclasses import dataclass

from src.utils import *
import matplotlib.pyplot as plt

@dataclass
class NeuralNetwotkConfig:
    """
    A class with variables for this Netural Network algorithm 
    """
    X_trn = None
    y_trn = None
    X_tst = None
    y_tst = None
    W = None
    params = None

class NeuralNetwotk:
    """
    A class with an attribute that sets the NeuralNetwotkConfig
    """
    def __init__(self):
        self.nn_config = NeuralNetwotkConfig()

    def set_params(self, params):
        """ Set the param in the class
        
        Expects:
            params: an object with the paramters to set in the class.
        Modifies:
            Nothing.
        Returns:
            Nothing.
        """
        self.nn_config.params = params    

    def init_theta_matrix(self, m, n):
        """ Returns W  as theta matrix with random values from -1 to 1
        
        Expects:
            m: an interger that represent the number ofrows.
            n: an interger that represent the number of columns.
        Modifies:
            Nothing.
        Returns:
            W: a m,n matrix with ramdon values from -1 to 1.
        """
        W =  np.random.uniform(-1,1,size=(m,n))
        return W

    def sigmoid(self, X):
        """ Returns the a matrix with sigmoid transformation
        
        Expects:
            X: a matrix with values to compute the sigmoid
        Modifies:
            It modifies X.
        Returns:
            X: a matrix with the sigmoid transformation
        """
        X = np.array(1 / (1 + np.exp( -1 * np.array(X) )))
        return X

    def cost_function(self, y_preds, y_trues, W_arr, n = 0, λ = 0):
        """ Returns the sigmoid transformation
        
        Expects:
            X: a matrix with values 
        Modifies:
            Nothing.
        Returns:
            It returns the cost function plus the regularization.
        """

        J = S = 0
        error_mat =   - np.multiply(y_trues, np.log(y_preds)) - np.multiply((1 - y_trues), np.log(1 - y_preds))
        errors = np.sum(error_mat, axis = 1)
        J = np.mean(errors)
        
        # squares of all errors - no bias term
        S = np.sum([np.sum( W_arr[i][ : , 1:W_arr[i].shape[1] ] **2) for i in range(len(W_arr))])
        S = λ/(2*n) *  S

        return J + S

    def create_thetas(self, architecture=[1,2,1]):
        """ Returns an array of theta matrixes 
        
        Expects:
            architecture: an array of intergers that represents the architecture 
                or the neural network
        Modifies:
            Nothing.
        Returns:
            W: an array of matrixes that represent the weiths of each layer of
                the neural network.
        """
        
        W = []
        for i in range(0, len(architecture) - 1):
            input_size = architecture[i]
            output_size = architecture[i+1]
            # adding the bias term to the first theta matrix
            W.append(self.init_theta_matrix(output_size, input_size + 1))

        return W   

    def forward_propagation(self, X, W_array, architecture=[3,2,1]):
        """ Returns an arrayof predictions and  acctivations
        
        Expects:
            X: a matrix with values.
            W_array: an array with the neural network activations.
            architecture: an array of intergers that represents the architecture
                or the neural network.
        Modifies:
            Nothing.
        Returns:
            y_preds: an array with the forward preditions.
            act_array: an array of theta matrixes.
        """

        total_layers = len(architecture)
        input_size = architecture[0]
        hidden_size = architecture[1]
        act_array = []
        
        ### First layer
        ### ============================================================================
        act_array.append(X)
        # concatenate the matrix and the new column along the second axis
        added_bias_term = np.concatenate((np.ones( (1,act_array[0].shape[0] )).T, act_array[0]), axis=1)
        act_array[0] = added_bias_term.T

        ### for intermiediate layers
        ### ============================================================================
        for k in range(1, total_layers - 1):
            W = W_array[k-1]
            # previous activation
            prev_act_mat = act_array[k-1]
            Z = W @ prev_act_mat
            
            # computing and appending the next activations
            a = self.sigmoid(Z)
            # concatenate the matrix and the new column along the second axis
            a = np.concatenate((np.ones( (1,a.shape[1] )), a), axis=0)
            act_array.append(a)
        
        ### for the last layer
        ### ============================================================================
        last_k= len(architecture)
        last_idx = len(architecture) - 2

        last_act_mat = act_array[len(act_array) - 1]
        last_W = W_array[last_idx]
        last_Z = last_W @ last_act_mat
        last_a = self.sigmoid(last_Z)
        act_array.append(last_a)
        y_preds = last_a.T
        
        return y_preds, act_array

    def back_propagation(self, X, y_trues, W_array, architecture=[1,2,1], λ=0, alpha=0):
        """ Returns the an array of activations and the regularized cost
        
        Expects:
            X: a matrix with values.
            y_trues: an array with the prediction of the forward propagation.
            W_array: an array with the neural network activations.
            architecture: an array of intergers that represents the architecture 
                or the neural network.
            λ: a float that represents the regularization parameter.
            alpha: a float that represents the neural netwokr learning rate.
        Modifies:
            Nothing.
        Returns:
            W_array: an array with the forward preditions
            cost: an array of theta matrixes 
        """

        last_layer_idx = len(architecture) - 1
        # init delta same length as thetas array
        delta_arr = [None for i in range(last_layer_idx )]
        D = [None for i in range(last_layer_idx )]
        P = [None for i in range(last_layer_idx )]
        
        # propagate and compute all the network outputs for all the instances
        # y_preds , act_arr = self.forward_propagation(X, W_array, architecture, mode = mode)
        y_preds , act_arr = self.forward_propagation(X, W_array, architecture)
        # compute the cost    
        cost = self.cost_function(y_preds, y_trues, W_array, n = X.shape[0], λ = λ)
        
        ### computes the delta values of all output neurons")
        ### ==========================================================================================
        delta_arr[last_layer_idx - 1] = (y_preds - y_trues).T
        
        #### for the hidden layers
        ### computes the delta values of all neurons in the hidden layers")
        ### ==========================================================================================
        for k in range(last_layer_idx - 1, 0, -1):
            delta = ( W_array[k].T @ delta_arr[k]) * act_arr[k] * (1- act_arr[k])
            # removing the bias term of all deltas
            delta= delta[1:]
            delta_arr[k-1] = delta

        #### updates gradients of the weights of each layer, based on the current training instance ")
        ### ==========================================================================================")
        for k in range(last_layer_idx - 1, -1, -1):
            D[k] = delta_arr[k] @ act_arr[k].T
        
        ### computes the final (regularized) gradients of the weights of each layer
        ### ==========================================================================================
        for k in range(last_layer_idx - 1, -1, -1):
            # computes regularizer, λ, of all non-bias weights and set its first column to be all zeros
            regularizers = λ * W_array[k]
            regularizers[:,0] = 0
            P[k] = regularizers
            # combines gradients w/ regularization terms; divides by #instances to obtain average gr
            D[k] = 1/len(y_trues) * (D[k] + P[k])
        
        ### updates the weights of each layer based on their corresponding gradients
        ### ==========================================================================================
        for k in range(last_layer_idx - 1, -1, -1):
            W_array[k] = W_array[k] - alpha * D[k] 
        
        return cost, W_array 

    def train_neural_network(self, X, y):
        """ Returns the updated weights (W) and the regularized cost
        
        Expects:
            X: a matrix that represnts training set. 
            y: an array that represents the true classes of matrix X.
        Modifies:
            Nothing.
        Returns:
            W_updated: an array theta matrixes that represents the updated 
                weights of neural network.
            cost: a float that represents the cost with regularization.
        """

        # retreive hyper-parameters from class itself
        architecture = self.nn_config.params["architecture"]
        λ = self.nn_config.params["lambda"]
        alpha = self.nn_config.params["learning_rate"]

        # Set the activation, weight
        W = self.create_thetas(architecture = architecture) 
        W = self.nn_config.W # load test from class

        # get better weights
        (cost, W_updated) = self.back_propagation(X, y, W, 
                                            architecture = architecture, 
                                            λ = λ, 
                                            alpha = alpha)

        return (W_updated , cost)
    
    def predict_neural_network(self, X):
        """ Returns the predictions of the neural network
        
        Expects:
            X: a matrix with values of the testing set.
        Modifies:
            Nothing.
        Returns:
            y_preds: an array with the predictions of the testing set.
        """

        # retreive variables from the class itself
        W = self.nn_config.W
        architecture = self.nn_config.params["architecture"]
        # y_preds , act_arr = self.forward_propagation(X, W, architecture, mode = mode)
        y_preds , act_arr = self.forward_propagation(X, W, architecture)
        
        return y_preds

    def grid_search_cv(self, dataset_obj, params, cv, mode=MODE.BREIF):
        """ Returns the best parameters combination with the highest accuracy

        Expects:
            dataset_obj: an object that contains: the dataframe, unique clases, number
                of classes, and dataframe information.
            params: an object with hyperparameters.
            cv: an integer with the number of cross-validation.
            mode: a string that represents the mode: how this method behave.
        Modifies:
            Nothing.
        Returns:
            best_params: an object with the best hyper-parameter settings.
        """
        
        best_params = {}
        best_accuracy = -99

        try:
            for archi_candidate in params["architectures"]:
                for learning_rate_candidate in params["learning_rates"]:
                    for lambda_candidate in params["lambdas"]:
                        for mini_batch_size_candidate in params["batch_sizes"]:
                            candidate_params = {
                                "architecture": archi_candidate,
                                "learning_rate": learning_rate_candidate,
                                "lambda": lambda_candidate,
                                "mini_batch_size": mini_batch_size_candidate,
                                "num_epochs": params["num_epochs"]
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

        k_accuracies = []   
        k_f1_scores = []    
        k_costs = []

        try:    
            for k in (range(int(cv))):

                df = dataset_obj["df"]
                classes = dataset_obj["classes"]
                num_classes = dataset_obj["num_classes"]
                dataset_info = dataset_obj["dataset_info"]

                df = df.sample(frac=1)
                
                (trn, tst) = train_test_split(df, k+1)
                (X_trn, y_trn, __) = attr_label_split(trn, num_classes)
                (X_tst, y_tst, y_trues) = attr_label_split(tst, num_classes)

                if mode == MODE.BREIF:
                    X_trn = X_trn[0:35,:]
                    y_trn = y_trn[0:35,:]
                    X_tst = X_tst[0:35,:]
                    y_tst = y_tst[0:35,:]
                    y_trues = y_trues[0:35]
                

                # get neural network configuration
                architecture = self.nn_config.params["architecture"]
                mini_batch_size = self.nn_config.params["mini_batch_size"]
                epochs = self.nn_config.params["num_epochs"] 

                ## setting the input and output neuron based on the given dataset
                architecture[0] = X_trn.shape[1]
                architecture[len(architecture) - 1] = y_trn.shape[1]
                
                # Set the activation, weight for TEST_CHAIN_W_EPOCH mode
                # W = self.create_thetas(architecture = architecture)   
                W = self.create_thetas(architecture = architecture)

                epochs_accuracy = []
                epochs_f1_scores = []
                epochs_cost = []
                
                ## Epochs - 1 epoch is completed when the Model sees the entire training set 
                for j in range(epochs):
                    # shuffling dataset
                    np.random.shuffle(X_trn)
                    # Metrics
                    acc = pre = rec = f1 = None
                    ## Minibatch varibles
                    init_cut = 0 
                    final_cut = mini_batch_size
                    total_instances = X_trn.shape[0]
                    num_batches = int(np.ceil(total_instances / mini_batch_size))
                    batches_cost = 0.0

                    ## Model starts drawing each batch to updte the W
                    for i in range(num_batches):
                        
                        if final_cut > total_instances:
                            final_cut = total_instances

                        # update the W matrix from the clas itself fo this particular epoch
                        self.nn_config.W = W 
                        # Training the model, calculating the error, and updatating the W for this batch
                        (W , cost ) = self.train_neural_network(X_trn[init_cut : final_cut, : ],
                                                        y_trn[init_cut : final_cut, : ])
                        ## update cuts
                        init_cut = final_cut
                        final_cut += mini_batch_size
                        
                        # accumulating the batch cost
                        batches_cost += cost

                    ## predic tst set 
                    y_preds = self.predict_neural_network(X_tst)
                    y_preds = get_y_preds_based_on_dataset(y_preds, dataset_info)

                    ## get the metrics
                    confusion_mat = create_confusion_matrix(y_trues, y_preds, classes)
                    (batches_acc, __, __, batches_f1) = get_multiclass_metrics(confusion_mat)

                    # store batch result for each epoch
                    epochs_accuracy.append(batches_acc)
                    epochs_f1_scores.append(batches_f1)
                    
                    # Store batches averrage cost for each epoc
                    epochs_cost.append(batches_cost/num_batches)
                
                # store average epochs_accuracy, epochs_f1_scores, and epochs_cost for each fold
                k_accuracies.append(np. mean(epochs_accuracy))
                k_f1_scores.append(np.mean(epochs_f1_scores))
                k_costs.append(np.mean(epochs_cost))
                
            return (np.mean(k_accuracies) )

        except Exception as e:
            raise CustomException(e, sys)
