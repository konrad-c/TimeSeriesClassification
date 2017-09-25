import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import classification_report

wafer_stretched = "NormalizationData/WaferClean/StretchedWafer.csv"
wafer_stretched_avg = "NormalizationData/WaferClean/StretchedAvgWafer.csv"
wafer_UCR = "NormalizationData/WaferClean/UCRWafer.csv"

def compare_ucr(filename_ucr, filename_raw, normalizer=None, sample_size=None, seed=8235416):
    fig, axes = plt.subplots(nrows=1,ncols=2, sharex=True, sharey=True)
    fig.set_size_inches(10, 7)
    data_ucr = np.array(pd.read_csv(filename_ucr))[:,1:]
    data_raw = np.array(pd.read_csv(filename_raw))[:,1:]
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
        axes[0].plot(data_ucr[i])
    for i in range(data_raw.shape[0]):
        axes[1].plot(data_raw[i])
    fig.tight_layout()
    fig.show()
    
def compare_closest(filenames_ucr, filenames_raw,normalizer = None, measure='euclidean',sample_size=3, seed=8235416):
    fig, axes = plt.subplots(nrows=1,ncols=2, sharex=True, sharey=True)
    fig.set_size_inches(10, 7)
    data_ucr = np.array(pd.read_csv(filenames_ucr))[:,1:]
    data_raw = np.array(pd.read_csv(filenames_raw))[:,1:]
    if normalizer is not None:
        scaler = normalizer()
        scaler.fit(data_raw)
        data_raw = scaler.transform(data_raw)
    np.random.seed(seed)
    data_ucr = data_ucr[np.random.choice(data_ucr.shape[0], size=sample_size, replace=False)]
    NN = NearestNeighbors(n_neighbors=1, metric='euclidean')
    NN.fit(data_raw)
    data_raw = data_raw[NN.kneighbors(data_ucr, return_distance=False).reshape(sample_size)]
    for i in range(data_ucr.shape[0]):
        axes[0].plot(data_ucr[i])
    for i in range(data_raw.shape[0]):
        axes[1].plot(data_raw[i])
    fig.tight_layout()
    fig.show()

def compare_closest_dtw(filenames_ucr, filenames_raw, normalizer = None,sample_size=3, seed=8235416):
    fig, axes = plt.subplots(nrows=3,ncols=2, sharex=True, sharey=True)
    fig.set_size_inches(10, 7)
    data_ucr = np.array(pd.read_csv(filenames_ucr))[:,1:]
    data_raw = np.array(pd.read_csv(filenames_raw))[:,1:]
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
        axes[0].plot(data_ucr[i])
    for i in range(data_raw.shape[0]):
        axes[1].plot(data_raw[i])
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
            analysis.NN_accuracy(filename_ucr,normalizer=norm,metric=metric,w=w,test_prop=0.86,seed=seed+r)
                for r in range(runs)]
        accuracy_comparison = [
            analysis.NN_accuracy(filename_comparison,normalizer=norm,metric=metric,w=w,test_prop=0.86,seed=seed+r)
                for r in range(runs)]
        print("Norm",str(n))
        print(accuracy_comparison)
        axes[n].boxplot([accuracy_ucr, accuracy_comparison], labels=["UCR", "Comparison"])
    fig.tight_layout()
    fig.show()
    
from wafer_norm import raw_wafer
#raw_wafer(filenames_normal, filenames_abnormal, stretch_length=None, rolling_average=False):
def gridsearch_length(lengths, runs, metric='euclidean', w=1, seed=2017, outfilename=None):
    DATA_PATH = "NormalizationData\\waferRaw\\"
    wafer_normal = [os.path.join(dp, f) for dp, dn, filenames in os.walk(DATA_PATH + "normal\\") for f in filenames]
    wafer_abnormal = [os.path.join(dp, f) for dp, dn, filenames in os.walk(DATA_PATH + "abnormal\\") for f in filenames]
    results_length = []
    results = []
    for l in lengths:
        classification,x = raw_wafer(wafer_normal, wafer_abnormal, stretch_length=l, rolling_average=False)
        classification = pd.factorize(classification)[0]
        classification = classification.reshape((classification.shape[0],1))
        for r in range(runs):
            results_length.append(l)
            acc = NN(np.concatenate((classification,x), axis=1),normalizer=analysis.RowStandardScaler,metric=metric,w=w,test_prop=0.86,seed=seed+r)
            results.append(acc)
    if outfilename is not None:
        out_file = open(outfilename, "w")
        out_file.write("TimeSeriesLength,Accuracy\n")
        for i in range(len(results_length)):
            out_file.write(str(results_length[i])+","+str(results[i])+"\n")
        out_file.close()
    fig, axes = plt.subplots(nrows=1,ncols=1, sharex=True, sharey=True)
    fig.set_size_inches(4, 4)
    axes.scatter(results_length, results)
    fig.tight_layout()
    fig.show()

lengths = [1,2,3,4]#[5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300]
gridsearch_length(lengths, 50, metric="euclidean",seed=666, outfilename="Results\\Wafer\\AccuracyLengthEuclideanSmall.csv")
#dimension = 'z'
#gridsearch_normal(wafer_UCR, wafer_stretched, 20, metric="euclidean",seed=2082)
#compare_closest(wafer_UCR, wafer_stretched,normalizer = analysis.RowStandardScaler, measure='euclidean',sample_size=2, seed=5)
#analysis.NN_accuracy(cricket_filenames_stretched['x'],normalizer=analysis.RowStandardScaler,metric="euclidean",seed=4)

#analysis(cricket_filenames_stretched['x'], RowStandardScaler, metric="euclidean", w=4, test_prop=0.5)
#analysis(cricket_filenames_UCR['x'], RowStandardScaler, metric="euclidean", w=4, test_prop=0.5)
#analysis(cricket_filenames_stretchedavg['x'], RowStandardScaler, metric="euclidean", w=4, test_prop=0.5)
#compare_ucr(gesture_filenames_UCR, gesture_filenames_stretched, normalizer=RowStandardScaler, sample_size=3,seed=122905)
#compare_closest(cricket_filenames_UCR, cricket_filenames_stretched, filenames_comparison=cricket_filenames_stretchedavg, normalizer=RowStandardScaler, sample_size=3,seed=11)
#compare_closest_dtw(cricket_filenames_UCR, cricket_filenames_stretched, normalizer=RowStandardScaler, sample_size=1,seed=11)


