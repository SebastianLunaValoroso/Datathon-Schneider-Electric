import numpy as np
import sklearn.datasets
import math
from abc import ABC, abstractmethod
import logging
import time
import multiprocessing
import pandas as pd
import pickle

logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG) # Capture all messages (DEBUG and higher)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING) #Maybe ERROR?
file_handler = logging.FileHandler("problems.log")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class RandomForestClassifier():
    """
    It includes a set of different decision trees, in this case,
    classifiers. It obtains a prediction by combining the ones of
    each decision tree by majority.
    """
    def __init__(self, num_trees, min_size, max_depth, ratio_samples, num_random_features, criterion, extra_trees):
        self._num_trees = num_trees
        self._min_size = min_size
        self._max_depth = max_depth
        self._ratio_samples = ratio_samples
        self._num_random_features = num_random_features
        self._criterion = criterion
        self._extra_trees = extra_trees #True or False
       
    def predict(self, X) -> int:
        """
        Returns an array of the predictions of diferent samples.
        """
        with multiprocessing.Pool() as pool:
            all_predictions = pool.map(self._predict_single, X)
        return np.array(all_predictions)

    def _predict_single(self, x):
        """
        Returns the prediction of a given set of features (from
        the same sample) by majority voting.
        """
        predictions = [tree.predict(x) for tree in self.decision_trees]
        return max(set(predictions), key=predictions.count)
   
    def fit(self, X, y):
        """
        It creates all of the decision trees.        
        """
        dataset = Dataset(X, y)
        self._make_decision_trees_multiprocessing(dataset)
        logger.debug('The decision trees have been created')

    def _make_decision_trees(self, dataset):
        self.decision_trees = []
        for _ in range(self._num_trees):
            subset = dataset.random_sampling(self._ratio_samples)
            tree = self._make_node(subset, 1)
            self.decision_trees.append(tree)
           
    def _make_decision_trees_multiprocessing(self, dataset):
        t1 = time.time()
        with multiprocessing.Pool() as pool:
            self.decision_trees = pool.starmap(self._create_tree, [(dataset, i) for i in range(self._num_trees)])
        t2 = time.time()
        logger.info('{} seconds per tree'.format((t2-t1)/self._num_trees))

    def _create_tree(self, dataset, nproc):
        subset = dataset.random_sampling(self._ratio_samples)
        tree = self._make_node(subset, 1)
        logger.debug(f'Process {nproc} finished creating a tree')
        return tree

    def _make_node(self, dataset, depth):
        if depth >= self._max_depth or dataset.num_samples <= self._min_size or len(np.unique(dataset.y)) == 1:
            node = self._make_leaf(dataset)
        else:
            node = self._make_parent_or_leaf(dataset, depth)
        logger.debug('A node has been created')
        return node

    def _make_leaf(self, dataset):
        logger.debug('The node created is a leaf')
        return Leaf(dataset.most_frequent_label())

    def _make_parent_or_leaf(self, dataset, depth):
        """
        Creates a leaf or a parent node, depending on the given dataset and depth.
        A leaf is generated if there is no possible split of the dataset.
        Otherwise, a parent is generated.
        """
        idx_features = np.random.choice(range(dataset.num_features), self._num_random_features, replace=False)
        best_feature_index, best_threshold, minimum_cost, best_split = self._best_split(idx_features, dataset)
        if best_split is None:
            return self._make_leaf(dataset)
   
        left_dataset, right_dataset = best_split
        assert left_dataset.num_samples > 0 or right_dataset.num_samples > 0
        if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
            return self._make_leaf(dataset)
        else:
            node = Parent(best_feature_index, best_threshold)
            node.left_child = self._make_node(left_dataset, depth + 1)
            node.right_child = self._make_node(right_dataset, depth + 1)
            return node

    def _best_split(self, idx_features, dataset):
        """
        Finds the best feature and threshold to split the dataset by some given features.
        It evaluates all possible splits using the given features and calculates
        the CART cost for each one. Then it selects the lowest cost split.
        """
        minimum_cost = np.inf
        best_feature_index = -1
        best_threshold = -1
        best_split = None
    
        if self._extra_trees:
            for idx in idx_features:
                min_value = np.min(dataset.X[:, idx])
                max_value = np.max(dataset.X[:, idx])
                if min_value == max_value:
                    continue  # Cnnot split with this feature
                threshold = np.random.uniform(min_value, max_value)
                left_dataset, right_dataset = dataset.split(idx, threshold)
                if len(left_dataset.X) == 0 or len(right_dataset.X) == 0:
                    continue
                cost = self._CART_cost(left_dataset, right_dataset)
                if cost < minimum_cost:
                    best_feature_index = idx
                    best_threshold = threshold
                    minimum_cost = cost
                    best_split = [left_dataset, right_dataset]
        else:
            for idx in idx_features:
                values = np.unique(dataset.X[:, idx])
                for val in values:
                    left_dataset, right_dataset = dataset.split(idx, val)
                    if len(left_dataset.X) == 0 or len(right_dataset.X) == 0:
                        continue
                    cost = self._CART_cost(left_dataset, right_dataset)
                    if cost < minimum_cost:
                        best_feature_index = idx
                        best_threshold = val
                        minimum_cost = cost
                        best_split = [left_dataset, right_dataset]
    
        return best_feature_index, best_threshold, minimum_cost, best_split

    def _gini(self, dataset):
        """
        Returns the Gini impurity of a given dataset.
        A 0 returned means all samples belong to the same class.
        """
        num_samples_per_class = dataset.dic_label_count()
        g = 1
        for n in num_samples_per_class.values():
            g -= (n/dataset.num_samples)**2
        logger.debug("Gini has been used to calculate the impurity")
        logger.debug("The Gini impurity of the given dataset is {}".format(g))
        return g
       
    def _CART_cost(self, left_dataset, right_dataset):
        """
        Calculates the Cart cost, to evaluate how well the classes are
        separated within each subset of the data.
        It is used to choose the best division while constructing each
        decision tree.
        """
        if self._criterion == 'gini':
            gl = self._gini(left_dataset)
            gr = self._gini(right_dataset)
        elif self._criterion == 'entropy':
            gl = self._entropy(left_dataset)
            gr = self._entropy(right_dataset)
        n_elements = left_dataset.num_samples + right_dataset.num_samples
        cost = gl*left_dataset.num_samples/n_elements + gr*right_dataset.num_samples/n_elements
        logger.debug("The Cart cost is {}".format(cost))
        return cost
       
    def _entropy(self, dataset):
        """
        Returns the entropy of a given dataset.
        A 0 returned means all samples belong to the same class.
        """
        num_samples_per_class = dataset.dic_label_count()
        e = 0
        for n in num_samples_per_class.values():
            if(n!=0):
                e -= (n/dataset.num_samples) * math.log(n/dataset.num_samples)
        logger.debug("Entropy has been used to calculate the impurity")
        logger.debug("The entropy of the given dataset is {}".format(e))
        return e

   
class Node(ABC):
    """Abstract Base Class for leaf's and parent's classes"""
    def __init__(self):
        self._right_child = self._left_child = None
       
    @abstractmethod
    def predict(self, x) -> int:
        pass

       
class Leaf(Node):
    """
    Includes the predicted class, representing the final decision
    after the splits.
    """
    def __init__(self, label):
        super().__init__()
        self._label = label

    def predict(self, x) -> int:
        return self._label


class Parent(Node):
    """
    Contains the decision rule used to split the data.
    """
    def __init__(self, feature_index, threshold):
        super().__init__()
        self._feature_index = feature_index
        self._threshold = threshold
   
    @property
    def left_child(self):
        return self._left_child    
   
    @left_child.setter
    def left_child(self, left_child):
        if not left_child:
            raise ValueError("left_child cannot be empty")
        self._left_child = left_child
   
    @property
    def right_child(self):
        return self._right_child
       
    @right_child.setter
    def right_child(self, right_child):
        if not right_child:
            raise ValueError("right_child cannot be empty")
        self._right_child = right_child
   
    def predict(self, x) -> int:
        if x[self._feature_index] < self._threshold:
            return self._left_child.predict(x)
        else:
            return self._right_child.predict(x)


class Dataset():
    """
    Contains the features and the groundtruth of the given samples
    of the Dataset used to train and test the classifier.
    """
    def __init__(self, X=None, y=None):
        self._X = X
        if len(y.shape) == 1:
            self._y = y.reshape(-1, 1)
        else:
            self._y = y
        self._num_samples = y.size
        if(len(X)==0):
            self._num_features=0
        else:
            self._num_features = X[0].size
       
    @property
    def X(self):
        return self._X
   
    @property
    def y(self):
        return self._y

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def num_features(self):
        return self._num_features
       
    def dic_label_count(self):
        """
        Returns a dictionary with all the labels and their count.
        """
        d = {}
        for i in range(self._num_samples):
            label = self._y[i, 0] #because y[i] is still a numpy array, not an int --> y[i,0] gets the value
            if label not in d.keys():
                d[label] = 1
            else:
                d[label] += 1
        logger.debug("The number of labels is {}".format(d))
        return d
   
    def random_sampling(self, ratio_samples):
        """
        Returns a new Dataset with some random samples of the actual one.
        The number of samples is given by "ratio_samples",
        that indicates the proportion of the actual we want to include.
        """
        assert 0.0 <= ratio_samples <= 1.0
        n_samples = int(self._num_samples * ratio_samples)
        assert n_samples > 0
        idxs = np.random.choice(self._num_samples, size=n_samples, replace=True)
        logger.debug("We have taken some samples of the dataset, to create a new one")
        return Dataset(self._X[idxs], self._y[idxs])
       
    def most_frequent_label(self):
        d = self.dic_label_count()
        logger.debug("The most frequent label has been computed.")
        return max(d, key=lambda k: d[k])
       
    def split(self, idx, threshold):
        """
        Divides the actual Dataset in two, given a threshold that is
        compared with one of the features of all samples.
        """
        idx_left = self._X[:, idx] < threshold
        idx_right = self._X[:, idx] >= threshold
        return Dataset(self._X[idx_left], self._y[idx_left]), Dataset(self._X[idx_right], self._y[idx_right])
       
def load_sonar():
    df = pd.read_csv('sonar_all_data.csv', header=None)
    X = df[df.columns[:-1]].to_numpy()
    y = df[df.columns[-1]].to_numpy(dtype=str)
    y = (y=='M').astype(int) # M = mine, R = rock
    return X, y

def load_MNIST():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
           
if __name__ == '__main__':
    which_dataset = "MNIST"
    
    if which_dataset == "MNIST":
        X_train, y_train, X_test, y_test = load_MNIST()
        num_samples, num_features = X_train.shape
        num_trees = 80
        min_size = 20
        max_depth = 20
        ratio_samples = 0.25
        num_random_features = 10
        criterion = 'gini'
        extra_trees = True    
    else:
        extra_trees = False
        if (which_dataset == "sonar"):
            X, y = load_sonar()
            num_trees = 5
            min_size = 10
            max_depth = 10
            ratio_samples = 0.7
            num_random_features = 14
            criterion = 'gini'
        elif which_dataset == "iris":  
            iris = sklearn.datasets.load_iris() # it's a dictionary
            X, y = iris.data, iris.target
            num_trees = 5
            min_size = 5
            max_depth = 10
            ratio_samples = 0.8
            num_random_features = int(np.sqrt(num_features))
            criterion = 'gini'
        
        ratio_train, ratio_test = 0.7, 0.3 # 70% train, 30% test
        num_samples, num_features = X.shape # 150, 4
        idx = np.random.permutation(range(num_samples)) # shuffle {0,1, ... 149} because samples come sorted by class
        num_samples_train = int(num_samples*ratio_train)
        num_samples_test = int(num_samples*ratio_test)
       
        idx_train = idx[:num_samples_train]
        idx_test = idx[num_samples_train : num_samples_train+num_samples_test]
       
        X_train, y_train = X[idx_train], y[idx_train]
        X_test, y_test = X[idx_test], y[idx_test]
   
    #file.write("num_trees,min_size,max_depth,ratio_samples,rand_feat,criterion->avg_accuracy\n")

    dataset1 = Dataset(X_train, y_train)
    dataset2 = Dataset(X_test, y_test)

    randomForest = RandomForestClassifier(num_trees, min_size, max_depth, ratio_samples, num_random_features, criterion, extra_trees)  
    randomForest.fit(X_train, y_train) #train
   
    ypred = randomForest.predict(X_test)
   
    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions/float(num_samples_test)
    print("num_trees: {}, min size: {}, max_depth: {}, ratio_samples {}, rand_feat: {}, {}".format(num_trees, min_size, max_depth, ratio_samples, num_random_features, criterion))
    print('accuracy {} %'.format(100*np.round(accuracy,decimals=7)))
   
    logger.info("{}: {},{},{},{},{},{}->{}\n".format(which_dataset,num_trees, min_size, max_depth, ratio_samples, num_random_features, criterion, 100*np.round(accuracy,decimals=7)))
