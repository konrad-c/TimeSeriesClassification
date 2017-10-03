import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import classification_report
import os
import time

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

import analysis
import matplotlib.pyplot as plt
from gesture_normalization import clean_data_stretched
def gridsearch_length(lengths, runs, metric='euclidean', w=1, seed=2017, outfilename=None):
    DATA_PATH = "NormalizationData\\OriginalDS-allRaw\\"
    gestures_data = [os.path.join(dp, f) for dp, dn, filenames in os.walk(DATA_PATH) for f in filenames]
    results_length = []
    results_x = []
    results_y = []
    results_z = []
    print("TimeSeriesLength,AccuracyX,AccuracyY,AccuracyZ,TimeTaken")
    for l in lengths:
        classification,x,y,z = clean_data_stretched(gestures_data, size=l, rolling_average=False)
        classification = pd.factorize(classification)[0]
        classification = classification.reshape((classification.shape[0],1))
        for r in range(runs):
            before = time.time()
            results_length.append(l)
            x_result = analysis.NN(np.concatenate((classification,x), axis=1),normalizer=analysis.RowStandardScaler,metric=metric,w=w,test_prop=0.8,seed=seed+r)
            results_x.append(x_result)
            y_result = analysis.NN(np.concatenate((classification,y), axis=1),normalizer=analysis.RowStandardScaler,metric=metric,w=w,test_prop=0.8,seed=seed+r)
            results_y.append(y_result)
            z_result = analysis.NN(np.concatenate((classification,z), axis=1),normalizer=analysis.RowStandardScaler,metric=metric,w=w,test_prop=0.8,seed=seed+r)
            results_z.append(z_result)
            print(str(l)+","+str(x_result)+","+str(y_result)+","+str(z_result),str(time.time()-before))
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

if __name__ == '__main__':
    lengths = [30,40]#[50,60,70,80,90,100,150,200,250,300,350,400,450,500,600,700,800,900,1000]
    gridsearch_length(lengths, 1, metric="DTW",seed=666, outfilename="Results\\Gestures\\TESTAccuracyLengthDTW_SMALL_MaxWindow.csv")



