import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import classification_report

class RowStandardScaler:
    def __init__(self):
        self.mean = 0
        self.std = 0
        
    def fit(self, X):
        self.mean = np.mean(X)#[X != 0])
        self.std = np.std(X)#[X != 0])
    
    def transform(self, X):
        return (X - self.mean)/self.std
        
class RowMinMaxScaler:
    def __init__(self):
        self.minimum = 0
        self.maximum = 0
        
    def fit(self, X):
        self.minimum = np.min(X)
        self.maximum = np.max(X)
    
    def transform(self, X):
        return (X - self.minimum)/(self.maximum - self.minimum)
###
### Sourced from http://alexminnaar.com/time-series-classification-and-clustering-with-python.html
###  
def DTWDistance(s1, s2,w=1):
    w = max(w, abs(len(s1)-len(s2)))
    DTW = np.ones((s1.shape[0]+1, s2.shape[0]+1), dtype=np.float)*np.inf
    DTW[0,0] = 0
    dists = np.power(s1.reshape(s1.shape[0],1) - s2, 2)

    for i in range(s1.shape[0]):
        dists[i] = np.power(s1[i] - s2, 2)
        for j in range(max(0, i-w), min(s2.shape[0], i+w)):
            DTW[i+1, j+1] = dists[i][j] + min(DTW[i, j+1],DTW[i+1, j], DTW[i, j])
    return np.sqrt(DTW[-1, -1])

###
### Sourced from http://alexminnaar.com/time-series-classification-and-clustering-with-python.html
###
def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    return np.sqrt(LB_sum)
    
def NN_DTW(x_train, y_train, x_test, y_test, w):
    predictions = []
    for i in range(x_test.shape[0]):
        test_i = x_test[i]
        min_dist = float('inf')
        closest_seq = None
        for j in range(x_train.shape[0]):
            train_j = x_train[j]
            if LB_Keogh(test_i, train_j, 5) < min_dist:
                dist = DTWDistance(test_i, train_j, w)
                if dist < min_dist:
                    min_dist = dist
                    closest_seq = j
        predictions.append(y_train[closest_seq])
    results = float(np.where(y_test == np.array(predictions))[0].shape[0])/float(np.array(predictions).shape[0])
    #print(classification_report(y_test, predictions))
    return results

def NN_accuracy(filename, normalizer=None, metric="euclidean", w=1, test_prop=0.5, seed=2082):
    np.random.seed(seed)
    data = pd.read_csv(filename, header=None)
    result = NN(np.array(data), normalizer, metric, w, test_prop, seed)
    return result

def NN(data, normalizer=None, metric="euclidean", w=1, test_prop=0.5, seed=2082):
    np.random.seed(seed)
    train, test = train_test_split(data, test_size=test_prop)
    y_train = np.array(train[:,0])
    x_train = np.array(train[:,1:])
    y_test= np.array(test[:,0])
    x_test = np.array(test[:,1:])
    if normalizer is not None:
        scaler = normalizer()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    if metric is "DTW":
        result = NN_DTW(x_train, y_train, x_test, y_test, w)
    else:
        classifier = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        #print(classification_report(y_test, predictions))
        result = float(np.where(y_test == predictions)[0].shape[0])/float(predictions.shape[0])
        #print(classification_report(y_test,predictions))
    return result





