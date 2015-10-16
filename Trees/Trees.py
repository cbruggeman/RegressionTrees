
from __future__ import division
import math

import numpy as np

from Metrics.metrics import RMSE, l2_gamma, l2_loss_gradient, SE




class DecisionTreeRegressor:
    def __init__(self,
                 loss = 'l2_loss',
                 min_samples_to_split = 5,
                 max_depth = None,
                 max_features = None,
                 sub_sample_size = None,
                 min_leaf_size = 1):
        self.loss = loss
        if self.loss == 'l2':
            self.loss_function = RMSE
        self.min_samples_to_split = min_samples_to_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.sub_sample_size = sub_sample_size
        self.min_leaf_size = min_leaf_size
        if self.min_samples_to_split <= 2 * self.min_leaf_size:
            self.min_samples_to_split = 2 * min_leaf_size + 1
        self.tree = Tree()


    def fit(self, X, y):
        tree_builder = RegressionTreeBuilder(min_samples_to_split = self.min_samples_to_split,
                                             max_depth = self.max_depth,
                                             max_features = self.max_features,
                                             sub_sample_size = self.sub_sample_size,
                                             min_leaf_size = self.min_leaf_size)
        self.tree = tree_builder.build_tree(X, y, remaining_depth = self.max_depth)

    def predict(self, X, y = None):
        return np.apply_along_axis(self.tree.get_prediction, axis = 1, arr = X)


class RegressionTreeBuilder:
    def __init__(self,
                 min_samples_to_split = 5,
                 max_depth = None, 
                 max_features = None,
                 sub_sample_size = None,
                 min_leaf_size = 1):
        self.tree = Tree()
        self.min_samples_to_split = max(min_samples_to_split,2)
        self.max_depth = max_depth
        self.max_features = max_features
        self.sub_sample_size = sub_sample_size
        self.min_leaf_size = min_leaf_size




    def build_tree(self,X,y, remaining_depth = None):
        if remaining_depth == None:
            remaining_depth = -1
        X = np.array(X)
        y = np.array(y)
        val = np.mean(y)
        if np.isnan(val):
            val = 0
        if len(y) < self.min_samples_to_split or remaining_depth == 0:
            return Tree(val = val)

        # Determine the subset of features (columns) to use for the split
        features = generate_subsets(X.shape[1], self.max_features)

        # Determine the subset of observations (rows, samples) to use for the split
        sub_sample = generate_subsets(X.shape[0], self.sub_sample_size)

        
        
        returned_feature, threshold_value, MSE_estimate = self.threshold_finder(X[sub_sample, :][:, features],y[sub_sample])
        threshold_feature = features[returned_feature]

        node = Tree()
        under_samples = X[:,threshold_feature] < threshold_value
        node.left = self.build_tree(X[under_samples], y[under_samples], remaining_depth - 1)
        node.right = self.build_tree(X[-under_samples], y[-under_samples], remaining_depth - 1)
        node.val = val
        node.threshold_value = threshold_value
        node.threshold_feature = threshold_feature

        return node


    def threshold_finder(self, X, y):
        """
        Finds the optimal threshold for the data X, y.

        Returns the feature and value of the threshold, as well as the resulting mean squared error.
        """
        threshold_feature, threshold_value = 0, 0
        full_SE = SE(np.mean(y),y)
        best_SE = full_SE
        try:
            num_feature = X.shape[1]
        except IndexError:
            X = X.reshape(1, X.shape[0])
            num_feature = X.shape[1]
        for feature in range(num_feature):
            column = X[:,feature]

            value, split_SE = self.threshold_single_feature(column, y)
            if split_SE < best_SE:
                threshold_feature, threshold_value, best_SE = feature, value, split_SE

        return threshold_feature, threshold_value, best_SE/len(y)


    def threshold_single_feature(self,x,y):
        """Picks a threshold value for x to minimize the sum of square errors for y.

        Returns the optimal threshold point, and the resulting sum of squared errors
        """
        x = np.array(x)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        n = len(y)
        left_of_split_sum = 0
        right_of_split_sum = sum(y)

        y_mean = np.mean(y)
        full_SE = SE(y_mean,y)

        best_k = self.min_leaf_size - 1
        best_SE = full_SE

        for k in range(self.min_leaf_size - 1, n - self.min_leaf_size):
            left_of_split_sum += y[k]
            right_of_split_sum -= y[k]
            left_prediction = left_of_split_sum / (k+1.)
            right_prediction = right_of_split_sum / (n-k-1.)

            split_SE = full_SE - (k+1)*(left_prediction - y_mean)**2 - (n-k-1) * (right_prediction - y_mean)**2

            if split_SE < best_SE:
                best_k, best_SE = k, split_SE

        
        threshold_point = (x[best_k]+x[best_k+1])/2. # This could likely be impoved, but shouldn't matter for large data sets

        return threshold_point, best_SE






class Tree:
    def __init__(self, 
                val = 0,
                left = None,
                right = None,
                threshold_feature = None,
                threshold_value = None):
        self.val = val
        self.left = left
        self.right = right
        self.threshold_value = threshold_value
        self.threshold_feature = threshold_feature

    def get_prediction(self,x):
        x = x.reshape(x.size)
        if not self.left:
            return self.val

        else:
            x_feature_val = x[self.threshold_feature]
            if x_feature_val >= self.threshold_value:
                return self.right.get_prediction(x)
            else:
                return self.left.get_prediction(x)




class RandomForestRegressor:
    def __init__(self,
                 n_estimators = 5,
                 min_samples_to_split = 5,
                 max_depth = None,
                 max_features = None,
                 sub_sample_size = None,
                 min_leaf_size = 1,
                 bootstrap_size = None,
                 verbose = 0):
        self.n_estimators = n_estimators
        self.min_samples_to_split = min_samples_to_split
        self.max_features = max_features
        self.max_depth = max_depth
        self.sub_sample_size = sub_sample_size
        self.min_leaf_size = min_leaf_size
        self.bootstrap_size = None
        self.verbose = verbose
        self.tree_list = [DecisionTreeRegressor(min_samples_to_split = min_samples_to_split,
                                                max_depth = max_depth,
                                                max_features = max_features,
                                                sub_sample_size = sub_sample_size,
                                                min_leaf_size = min_leaf_size,) for _ in range(n_estimators)]


    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        for k, tree in enumerate(self.tree_list):
            if self.verbose >= 1:
                print "Training tree: ",k
            bootstrap_sample = generate_subsets(X.shape[0], self.bootstrap_size)
            tree.fit(X[bootstrap_sample,:],y[bootstrap_sample])

    def predict(self, X, y = None):
        return np.array([tree.predict(X) for tree in self.tree_list]).T.mean(axis = 1)



class GradientBoostingRegressor:

    def __init__(self,
                 n_estimators = 100,
                 learning_rate = 0.1,
                 max_depth = 3,
                 objective = 'l2',
                 gradient = None,
                 gamma_func = l2_gamma,
                 min_samples_to_split = 5,
                 max_features = None,
                 sub_sample_size = None,
                 min_leaf_size = 1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_to_split = min_samples_to_split
        self.max_features = max_features
        self.sub_sample_size = sub_sample_size
        self.min_leaf_size = min_leaf_size
        self.gammas = [0 for _ in range(n_estimators)]

        self.tree_list = [DecisionTreeRegressor(min_samples_to_split = min_samples_to_split,
                                                max_depth = max_depth,
                                                max_features = max_features,
                                                sub_sample_size = sub_sample_size,
                                                min_leaf_size = min_leaf_size,) for _ in range(n_estimators)]

        self.objective = objective
        if objective == 'l2':
            self.objective = RMSE

        self.gradient = gradient
        if gradient is None:
            self.gradient = l2_loss_gradient

        self.gamma_func = gamma_func

    def fit(self,X,y):
        residuals = y
        prediction = np.zeros(len(y))

        for k, tree in enumerate(self.tree_list):
            tree.fit(X,residuals)
            h = tree.predict(X)
            self.gammas[k] = self.gamma_func(y, prediction, h)
            prediction = prediction + self.learning_rate * self.gammas[k] * h

            residuals = self.gradient(prediction, y)

    def predict(self, X, y = None, num_trees = None, return_list = False):
        prediction = np.zeros(X.shape[0])
        list_of_prediction = []
        if num_trees == None:
            num_trees = self.n_estimators - 1
        num_trees = min(self.n_estimators - 1, num_trees)
        for gamma, tree in zip(self.gammas, self.tree_list)[:num_trees + 1]:
            prediction += self.learning_rate * gamma * tree.predict(X)
            if return_list:
                list_of_prediction.append(prediction.copy())

        if list_of_prediction:
            return list_of_prediction
        else:
            return prediction


    def score(self, X, y, loss = RMSE, num_trees = None):
        return loss(self.predict(X), y)

    def score_by_depth(self, X, y, loss = RMSE):
        return [loss(prediction, y) for prediction in self.predict(X, return_list = True)]


def generate_subsets(a, size):
    if type(size) == float and 0 < size < 1:
        num = int(math.floor(size*a))
        subset = np.random.choice(a,num,replace = False)
    elif type(size) == int:
        subset = np.random.choice(a,size,replace = False)
    else:
        subset = range(a)

    return subset