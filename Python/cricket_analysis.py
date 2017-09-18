import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import classification_report

cricket_filenames = {
    'x': "NormalizationData/CricketClean/",
    'y': "NormalizationData/CricketClean/",
    'z': "NormalizationData/CricketClean/",
}

cricket_filenames_stretched = {
    'x': "NormalizationData/CricketClean/StretchedCricketX.csv",
    'y': "NormalizationData/CricketClean/StretchedCricketY.csv",
    'z': "NormalizationData/CricketClean/StretchedCricketZ.csv"
}

cricket_filenames_stretchedavg = {
    'x': "NormalizationData/CricketClean/StretchedAvgCricketX.csv",
    'y': "NormalizationData/CricketClean/StretchedAvgCricketY.csv",
    'z': "NormalizationData/CricketClean/StretchedAvgCricketZ.csv"
}

cricket_filenames_UCR = {
    'x': "UCRData_long/Cricket_X/Cricket_X_TEST",
    'y': "UCRData_long/Cricket_Y/Cricket_Y_TEST",
    'z': "UCRData_long/Cricket_Z/Cricket_Z_TEST"
}

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
        data_ucr = data_ucr[np.random.choice(data_ucr.shape[0], size=sample_size, replace=False)]
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

def compare_closest_dtw(filenames_ucr, filenames_raw, normalizer = None,sample_size=3, seed=8235416):
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
        data_ucr = data_ucr[np.random.choice(data_ucr.shape[0], size=sample_size, replace=False)]
        NN = NearestNeighbors(n_neighbors=1, metric=DTWDistance)
        NN.fit(data_raw)
        data_raw = data_raw[NN.kneighbors(data_ucr, return_distance=False).reshape(sample_size)]
        for i in range(data_ucr.shape[0]):
            axes[d,0].plot(data_ucr[i])
        for i in range(data_raw.shape[0]):
            axes[d,1].plot(data_raw[i])
    fig.tight_layout()
    fig.show()

import analysis
import matplotlib.pyplot as plt
def gridsearch_normal(filename_ucr, filename_comparison, runs, metric='euclidean', w=1, seed=2017):
    fig, axes = plt.subplots(nrows=3,ncols=1, sharex=True, sharey=True)
    fig.set_size_inches(3, 7)
    normalizers = [
        analysis.RowStandardScaler,
        analysis.RowMinMaxScaler,
        None
    ]
    for n,norm in enumerate(normalizers):
        accuracy_ucr = [
            analysis.NN_accuracy(filename_ucr,normalizer=norm,metric=metric,w=w,test_prop=0.95,seed=seed+r)
                for r in range(runs)]
        accuracy_comparison = [
            analysis.NN_accuracy(filename_comparison,normalizer=norm,metric=metric,w=w,test_prop=0.95,seed=seed+r)
                for r in range(runs)]
        print("Norm",str(n))
        print(accuracy_comparison)
        axes[n].boxplot([accuracy_ucr, accuracy_comparison], labels=["UCR", "Comparison"])
    fig.tight_layout()
    fig.show()
    
dimension = 'x'
gridsearch_normal(cricket_filenames_UCR[dimension], cricket_filenames_stretched[dimension], 1, metric="DTW",seed=1)
#analysis.NN_accuracy(cricket_filenames_stretched['x'],normalizer=analysis.RowStandardScaler,metric="euclidean",seed=4)

#analysis(cricket_filenames_stretched['x'], RowStandardScaler, metric="euclidean", w=4, test_prop=0.5)
#analysis(cricket_filenames_UCR['x'], RowStandardScaler, metric="euclidean", w=4, test_prop=0.5)
#analysis(cricket_filenames_stretchedavg['x'], RowStandardScaler, metric="euclidean", w=4, test_prop=0.5)
#compare_ucr(gesture_filenames_UCR, gesture_filenames_stretched, normalizer=RowStandardScaler, sample_size=3,seed=122905)
#compare_closest(cricket_filenames_UCR, cricket_filenames_stretched, filenames_comparison=cricket_filenames_stretchedavg, normalizer=RowStandardScaler, sample_size=3,seed=11)
#compare_closest_dtw(cricket_filenames_UCR, cricket_filenames_stretched, normalizer=RowStandardScaler, sample_size=1,seed=11)


