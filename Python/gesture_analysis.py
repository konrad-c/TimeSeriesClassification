import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import classification_report

gesture_filenames = {
    'x': "NormalizationData/GesturesRaw/GesturesX.csv",
    'y': "NormalizationData/GesturesRaw/GesturesY.csv",
    'z': "NormalizationData/GesturesRaw/GesturesZ.csv"
}

gesture_filenames_stretched = {
    'x': "NormalizationData/GesturesRaw/StretchedGesturesX.csv",
    'y': "NormalizationData/GesturesRaw/StretchedGesturesY.csv",
    'z': "NormalizationData/GesturesRaw/StretchedGesturesZ.csv"
}

gesture_filenames_stretchedavg = {
    'x': "NormalizationData/GesturesRaw/StretchedAvgGesturesX.csv",
    'y': "NormalizationData/GesturesRaw/StretchedAvgGesturesY.csv",
    'z': "NormalizationData/GesturesRaw/StretchedAvgGesturesZ.csv"
}

gesture_filenames_timenormal = {
    'x': "NormalizationData/GesturesRaw/TimeNormalGesturesX.csv",
    'y': "NormalizationData/GesturesRaw/TimeNormalGesturesY.csv",
    'z': "NormalizationData/GesturesRaw/TimeNormalGesturesZ.csv"
}

gesture_filenames_timenormalavg = {
    'x': "NormalizationData/GesturesRaw/TimeNormalAvgGesturesX.csv",
    'y': "NormalizationData/GesturesRaw/TimeNormalAvgGesturesY.csv",
    'z': "NormalizationData/GesturesRaw/TimeNormalAvgGesturesZ.csv"
}

gesture_filenames_UCR = {
    'x': "UCRData_long/uWaveGestureLibrary_X/uWaveGestureLibrary_X_TEST",
    'y': "UCRData_long/uWaveGestureLibrary_Y/uWaveGestureLibrary_Y_TEST",
    'z': "UCRData_long/uWaveGestureLibrary_Z/uWaveGestureLibrary_Z_TEST"
}

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
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])
    
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
    
###
### Based on source from http://alexminnaar.com/time-series-classification-and-clustering-with-python.html
###
def NN_speedup(x_train, y_train, x_test, y_test, w):
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
    return classification_report(y_test, predictions)

def analysis(filename, normalizer, metric="DTW", w=1, test_prop=0.5):
    data = pd.read_csv(filename, header=None)
    train, test = train_test_split(data, test_size=test_prop)
    y_train = np.array(train[0])
    x_train = np.array(train.drop(0, axis=1))
    y_test= np.array(test[0])
    x_test = np.array(test.drop(0, axis=1))
    if normalizer is not None:
        scaler = normalizer()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    
    if metric is "DTW":
        print(NN_speedup(x_train, y_train, x_test, y_test, w))
    else:
        classifier = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        print(classification_report(y_test, predictions))

def compare_ucr(filenames_ucr, filenames_raw, normalizer=None, sample_size=None, seed=8235416):
    fig, axes = plt.subplots(nrows=3,ncols=2, sharex=True, sharey=True)
    fig.set_size_inches(10, 7)
    for d,dim in enumerate(['x','y','z']):
        data_ucr = np.array(pd.read_csv(filenames_ucr[dim]))[:,1:]
        data_raw = np.array(pd.read_csv(filenames_raw[dim]))[:,1:]
        if normalizer is not None:
            scaler = normalizer()
            scaler.fit(data_raw)
            data_raw = scaler.transform(data_raw)
        if sample_size is not None:
            np.random.seed(seed)
            data_ucr = data_ucr[np.random.choice(data_ucr.shape[0], size=sample_size ,replace=True)]
            np.random.seed(seed)
            data_raw = data_raw[np.random.choice(data_raw.shape[0], size=sample_size ,replace=True)]
        for i in range(data_ucr.shape[0]):
            axes[d,0].plot(data_ucr[i])
        for i in range(data_raw.shape[0]):
            axes[d,1].plot(data_raw[i])
    fig.tight_layout()
    fig.show()
    
def compare_closest(filenames_ucr, filenames_raw,normalizer = None, measure='euclidean',sample_size=3, seed=8235416, filenames_comparison=None):
    fig, axes = plt.subplots(nrows=3,ncols=(2 if filenames_comparison is None else 3), sharex=True, sharey=True)
    fig.set_size_inches(10, 7)
    for d,dim in enumerate(['x','y','z']):
        data_ucr = np.array(pd.read_csv(filenames_ucr[dim]))[:,1:]
        data_raw = np.array(pd.read_csv(filenames_raw[dim]))[:,1:]
        if filenames_comparison is not None:
            data_comp = np.array(pd.read_csv(filenames_comparison[dim]))[:,1:]
        if normalizer is not None:
            scaler = normalizer()
            scaler.fit(data_raw)
            data_raw = scaler.transform(data_raw)
            if filenames_comparison is not None:
                scaler_comp = normalizer()
                scaler_comp.fit(data_comp)
                data_comp = scaler_comp.transform(data_comp)
        np.random.seed(seed)
        data_ucr = data_ucr[np.random.choice(data_ucr.shape[0], size=sample_size ,replace=True)]
        NN = NearestNeighbors(n_neighbors=1, metric='euclidean')
        NN.fit(data_raw)
        data_raw = data_raw[NN.kneighbors(data_ucr, return_distance=False).reshape(sample_size)]
        if filenames_comparison is not None:
            data_comp = data_comp[NN.kneighbors(data_ucr, return_distance=False).reshape(sample_size)]
        for i in range(data_ucr.shape[0]):
            axes[d,0].plot(data_ucr[i])
        for i in range(data_raw.shape[0]):
            axes[d,1].plot(data_raw[i])
        if filenames_comparison is not None:
            for i in range(data_comp.shape[0]):
                axes[d,2].plot(data_comp[i])
    fig.tight_layout()
    fig.show()

def compare_closest_dtw(filenames_ucr, filenames_raw,normalizer = None,sample_size=3, seed=8235416):
    fig, axes = plt.subplots(nrows=3,ncols=2, sharex=True, sharey=True)
    fig.set_size_inches(10, 7)
    for d,dim in enumerate(['x','y','z']):
        data_ucr = np.array(pd.read_csv(filenames_ucr[dim]))[:,1:]
        data_raw = np.array(pd.read_csv(filenames_raw[dim]))[:,1:]
        if normalizer is not None:
            scaler = normalizer()
            scaler.fit(data_raw)
            data_raw = scaler.transform(data_raw)
        np.random.seed(seed)
        data_ucr = data_ucr[np.random.choice(data_ucr.shape[0], size=sample_size ,replace=True)]
        NN = NearestNeighbors(n_neighbors=1, metric=)
        NN.fit(data_raw)
        data_raw = data_raw[NN.kneighbors(data_ucr, return_distance=False).reshape(sample_size)]
        for i in range(data_ucr.shape[0]):
            axes[d,0].plot(data_ucr[i])
        for i in range(data_raw.shape[0]):
            axes[d,1].plot(data_raw[i])
    fig.tight_layout()
    fig.show()

analysis(gesture_filenames_stretched['x'], RowStandardScaler, metric="euclidean", w=4, test_prop=0.8)
#compare_ucr(gesture_filenames_UCR, gesture_filenames_stretched, normalizer=RowStandardScaler, sample_size=3,seed=122905)
compare_closest(gesture_filenames_UCR, gesture_filenames_stretchedavg, filenames_comparison=gesture_filenames_stretched, normalizer=RowStandardScaler, sample_size=3,seed=5)

data_raw = np.array(pd.read_csv(filenames_raw['x']))[:,1:]#

classification,x,y,z
sample = np.random.choice(x.shape[0], size=sample_size ,replace=True)
x_sample = x[sample]

