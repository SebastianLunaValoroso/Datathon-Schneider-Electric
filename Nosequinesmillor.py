import numpy as np
import sklearn.datasets
import random
import math
from abc import ABC, abstractmethod
import logging
import time
import multiprocessing
        

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
    def __init__(self, num_trees, min_size, max_depth, ratio_samples, num_random_features, criterion):
        self._num_trees = num_trees
        self._min_size = min_size
        self._max_depth = max_depth
        self._ratio_samples = ratio_samples
        self._num_random_features = num_random_features
        self._criterion = criterion
    """
    def predict(self, X) -> int:
        ypred = []
        for x in X:
            predictions = [root.predict(x) for root in self.decision_trees]
            #majority voting
            ypred.append(max(set(predictions), key=predictions.count))
            logger.debug('The prediction is {}'.format(np.array(ypred)))
        return np.array(ypred)
    """
    def predict(self, X) -> int:
        with multiprocessing.Pool() as pool:
            all_predictions = pool.map(self._predict_single, X)
        return np.array(all_predictions)

    def _predict_single(self, x):
        predictions = [tree.predict(x) for tree in self.decision_trees]
        return max(set(predictions), key=predictions.count)
    
    def fit(self, X, y):
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
        idx_features = np.random.choice(range(dataset.num_features), self._num_random_features, replace=False)
        best_feature_index, best_threshold, minimum_cost, best_split = self._best_split(idx_features, dataset)
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
        best_feature_index, best_threshold, minimum_cost, best_split = np.Inf, np.Inf, np.Inf, None
        for idx in idx_features:
            values = np.unique(dataset.X[:, idx])
            for val in values:
                left_dataset, right_dataset = dataset.split(idx, val)
                if len(left_dataset.X) == 0 or len(right_dataset.X) == 0:
                    continue
                cost = self._CART_cost(left_dataset, right_dataset)
                if cost < minimum_cost:
                    best_feature_index, best_threshold, minimum_cost, best_split = idx, val, cost, [left_dataset, right_dataset]
       
        return best_feature_index, best_threshold, minimum_cost, best_split

    def _gini(self, dataset):
        num_samples_per_class = dataset.dic_label_count()
        g = 1
        for n in num_samples_per_class.values():
            g -= (n/dataset.num_samples)**2
        logger.debug("Gini has been used to calculate the impurity")
        logger.debug("The Gini impurity of the given dataset is {}".format(g))
        return g
       
    def _CART_cost(self, left_dataset, right_dataset):
        if self._criterion == 'gini':
            gl = self._gini(left_dataset)
            gr = self._gini(right_dataset)
        elif self._criterion == 'entropy':
            gl = self._entropy(left_dataset)
            gr = self._entropy(right_dataset)
        n_elements = left_dataset.num_samples + right_dataset.num_samples
        cost = gl*left_dataset.num_samples/n_elements + gr*right_dataset.num_samples/n_elements
        # Implement the CART cost function here
        logger.debug("The Cart cost is {}".format(cost))
        return cost
       
    def _entropy(self, dataset):
        """ADD TO PLANT UML"""
        num_samples_per_class = dataset.dic_label_count()
        e = 0
        for n in num_samples_per_class.values():
            if(n!=0):
                e -= (n/dataset.num_samples) * math.log(n/dataset.num_samples)
        logger.debug("Entropy has been used to calculate the impurity")
        logger.debug("The entropy of the given dataset is {}".format(e))
        return e

class Node(ABC):
    def __init__(self):
        self._right_child = self._left_child = None
       
    @abstractmethod
    def predict(self, x) -> int:
        pass

       
class Leaf(Node):
    def __init__(self, label):
        super().__init__()
        self._label = label

    def predict(self, x) -> int:
        return self._label


class Parent(Node):
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
   
   
    """
    @property
    def feature_index(self):
        return self._feature_index
       
    @property
    def threshold(self):
        return self._threshold
    """
   
    def predict(self, x) -> int:
        if x[self._feature_index] < self._threshold:
            return self._left_child.predict(x)
        else:
            return self._right_child.predict(x)


class Dataset():
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
        n_samples = int(self._num_samples * ratio_samples)
        idxs = np.random.choice(self._num_samples, size=n_samples, replace=True)
        subset_X = self._X[idxs]
        subset_y = self._y[idxs]
        """
        i_rand = random.randint(0, self._num_samples-1)
        subset_X = self._X[i_rand]
        subset_y = self._y[i_rand]
       
        for i in range(n_samples):
            i_rand = random.randint(0, self._num_samples)
            subset_X = np.vstack((subset_X, self._X[i_rand]))
            subset_y = np.vstack((subset_y, self._y[i_rand]))
        """
        logger.debug("We have taken some samples of the dataset, to create a new one")
        return Dataset(subset_X, subset_y)
       
    def most_frequent_label(self):
        d = self.dic_label_count()
        #max_label = max(d.values())
        #return d[ d.keys()[d.values().index(max_label)] ]
        logger.debug("The most frequent label has been computed.")
        return max(d, key=lambda k: d[k])
         
    def split(self, idx, val):
        left_X = np.empty((0, self._num_features))
        right_X = np.empty((0,self._num_features))
       
        #Donem per fet que el vector de label és una columna.
        left_y = np.empty((0, 1))
        right_y = np.empty((0,1))
       
        for i in range(self._num_samples):
            if self._X[i, idx] <= val:
                left_X = np.vstack((left_X, self._X[i]))
                left_y = np.vstack((left_y, self._y[i]))
            else:
                right_X = np.vstack((right_X, self._X[i]))
                right_y = np.vstack((right_y, self._y[i]))
        return Dataset(left_X, left_y), Dataset(right_X, right_y)
       
    def divide_dataset(self, ratio):
        arr = np.arange(self._num_samples)
        np.random.shuffle(arr)
       
        first = int(self._num_samples*ratio)
        X1 = np.empty((0, self._num_features))
        y1 = np.empty((0,1))
        for i in range(first):
            X1 = np.vstack((X1, self._X[arr[i]]))
            y1 = np.vstack((y1, self._y[arr[i]]))
        X2 = np.empty((0, self._num_features))
        y2 = np.empty((0,1))
        for i in range(first, self._num_samples):
            X2 = np.vstack((X2, self._X[arr[i]]))
            y2 = np.vstack((y2, self._y[arr[i]]))
        return Dataset(X1, y1), Dataset(X2, y2)
       
if __name__ == '__main__':
    """
    X_llista = []
    Y_llista = []
    with open("sonar.all-data.csv") as file:
        for line in file:
            linia = line.split(',')
            X_llista.append(linia[:-1])
            Y_llista.append(linia[-1])
    
    X = np.array(X_llista)
    Y = np.array(Y_llista)
    """
    iris = sklearn.datasets.load_iris() # it's a dictionary
    X, y = iris.data, iris.target
    ratio_train, ratio_test = 0.7, 0.3 # 70% train, 30% test
    num_samples, num_features = X.shape # 150, 4
    idx = np.random.permutation(range(num_samples)) # shuffle {0,1, ... 149} because samples come sorted by class!
    num_samples_train = int(num_samples*ratio_train)
    num_samples_test = int(num_samples*ratio_test)
    
    idx_train = idx[:num_samples_train]
    idx_test = idx[num_samples_train : num_samples_train+num_samples_test]
    
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    dataset1 = Dataset(X_train, y_train)
    dataset2 = Dataset(X_test, y_test)
   
    num_trees = 10
    min_size = 5
    max_depth = 5
    ratio_samples = 0.8
    num_random_features = int(np.sqrt(num_features))
    criterion = "gini"
   
    randomForest = RandomForestClassifier(num_trees, min_size, max_depth, ratio_samples, num_random_features, criterion)   
    randomForest.fit(X_train, y_train) #train
   
    ypred = randomForest.predict(X_test)
    
    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions/float(num_samples_test)
    print('accuracy {} %'.format(100*np.round(accuracy,decimals=2)))
   
"""
import sklearn
if __name__ == '__main__':

mnist =sklearn.datasets.fetch_mldata('mnist_784')
x = mnist.datay = mnist.target
"""