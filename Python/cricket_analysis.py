import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import classification_report
import multiprocessing as mp
import os

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
    'x': "NormalizationData/CricketClean/UCRCricketX.csv",
    'y': "NormalizationData/CricketClean/UCRCricketY.csv",
    'z': "NormalizationData/CricketClean/UCRCricketZ.csv"
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
            analysis.NN_accuracy(filename_ucr,normalizer=norm,metric=metric,w=w,test_prop=0.5,seed=seed+r)
                for r in range(runs)]
        accuracy_comparison = [
            analysis.NN_accuracy(filename_comparison,normalizer=norm,metric=metric,w=w,test_prop=0.5,seed=seed+r)
                for r in range(runs)]
        print("Norm",str(n))
        print(accuracy_comparison)
        axes[n].boxplot([accuracy_ucr, accuracy_comparison], labels=["UCR", "Comparison"])
    fig.tight_layout()
    fig.show()
    
from cricket_normalization import clean_data_stretched
def gridsearch_length(lengths, runs, metric='euclidean', w=1, seed=2017, outfilename=None):
    DATA_PATH = "NormalizationData\\CricketDataRaw\\"
    cricket_data = [os.path.join(dp, f) for dp, dn, filenames in os.walk(DATA_PATH) for f in filenames]
    results_length = []
    results_x = []
    results_y = []
    results_z = []
    for l in lengths:
        classification,x,y,z = clean_data_stretched(cricket_data, size=l, rolling_average=False)
        classification = pd.factorize(classification)[0]
        classification = classification.reshape((classification.shape[0],1))
        for r in range(runs):
            results_length.append(l)
            x_result = analysis.NN(np.concatenate((classification,x), axis=1),normalizer=analysis.RowStandardScaler,metric=metric,w=w,test_prop=0.5,seed=seed+r)
            results_x.append(x_result)
            y_result = analysis.NN(np.concatenate((classification,y), axis=1),normalizer=analysis.RowStandardScaler,metric=metric,w=w,test_prop=0.5,seed=seed+r)
            results_y.append(y_result)
            z_result = analysis.NN(np.concatenate((classification,z), axis=1),normalizer=analysis.RowStandardScaler,metric=metric,w=w,test_prop=0.5,seed=seed+r)
            results_z.append(z_result)
    if outfilename is not None:
        out_file = open(outfilename, "w")
        out_file.write("TimeSeriesLength,AccuracyX,AccuracyY,AccuracyZ\n")
        for i in range(len(results_length)):
            out_file.write(str(results_length[i])+","+str(results_x[i])+","+str(results_y[i])+","+str(results_z[i])+"\n")
        out_file.close()
    #fig, axes = plt.subplots(nrows=3,ncols=1, sharex=True, sharey=True)
    #fig.set_size_inches(3, 7)
    #axes[0].scatter(results_length, results_x)
    #axes[1].scatter(results_length, results_y)
    #axes[2].scatter(results_length, results_z)
    #fig.tight_layout()
    #fig.show()

def get_data(metric, out_filename, runs, lengths):
    gridsearch_length(lengths, runs, metric=metric,seed=666, outfilename="Results\\Cricket\\"+out_filename)

def get_data_parallel(metric, out_filename, runs, window, lengths):
    num_proc = mp.cpu_count()
    assert len(lengths) % 8 == 0, "Length must be multiple of " + str(num_proc)
    jobs_per_proc = len(lengths)//num_proc
    jobs = []
    for i in range(num_proc):
        #gridsearch_length(lengths[jobs_per_proc*i:jobs_per_proc*(i+1)], runs, metric=metric, seed=666, outfilename="Results\\Cricket\\Process_"+str(i)+out_filename)
        p = mp.Process(target=gridsearch_length, args=(lengths[jobs_per_proc*i:jobs_per_proc*(i+1)], runs, metric, window, 666, "Results\\Cricket\\Process_"+str(i)+out_filename,))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
        print(str(j.name)+".exitcode = "+str(j.exitcode))

if __name__ == '__main__':
    get_data_parallel("DTW", "AccuracyLengthDTW_MaxWindow.csv", 8, None, [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800])
#get_data("euclidean", "AccuracyLengthEuclidean.csv", 100, [5,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,1500,3000])
#get_data("DTW", "AccuracyLengthDTW.csv", 5, [50,100,200,300,400,500,600,800])

#dimension = 'z'
#gridsearch_normal(cricket_filenames_UCR[dimension], cricket_filenames_stretched[dimension], 100, metric="euclidean",seed=2082)

#analysis.NN_accuracy(cricket_filenames_stretched['x'],normalizer=analysis.RowStandardScaler,metric="euclidean",seed=4)

#analysis(cricket_filenames_stretched['x'], RowStandardScaler, metric="euclidean", w=4, test_prop=0.5)
#analysis(cricket_filenames_UCR['x'], RowStandardScaler, metric="euclidean", w=4, test_prop=0.5)
#analysis(cricket_filenames_stretchedavg['x'], RowStandardScaler, metric="euclidean", w=4, test_prop=0.5)
#compare_ucr(gesture_filenames_UCR, gesture_filenames_stretched, normalizer=RowStandardScaler, sample_size=3,seed=122905)
#compare_closest(cricket_filenames_UCR, cricket_filenames_stretched, filenames_comparison=cricket_filenames_stretchedavg, normalizer=RowStandardScaler, sample_size=3,seed=11)
#compare_closest_dtw(cricket_filenames_UCR, cricket_filenames_stretched, normalizer=RowStandardScaler, sample_size=1,seed=11)


