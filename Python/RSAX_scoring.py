import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ultrafast_shapelets as shapelets
import pysax
import timeit
import os

np.set_printoptions(suppress=True)

# Classification
#from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

def train_scoring(T_train, label_train, T_test, label_test, seed=-1, dtw=False):
    """
    Run timer and plot time complexity/accuracy
    """
    before_dataset = timeit.time.time()
    shapelet_nums = np.arange(130,181,10)
    raw_num = []
    time_arr = []
    accuracy = []
    data_all = None
    for i in shapelet_nums:
        acc = 0
        t = 0
        # Check time
        before = timeit.time.time()
        T_train_features, T_shapelets, info = shapelets.motif_feature_approx(T_train, label_train, i, i, reduced=True, seed=seed, dtw=dtw)
        # Check accuracy
        T_test_features = shapelets.ts_features_from_shapelets(T_test, T_train, np.array(T_shapelets), dtw=dtw)
        # Predict
        model_svc = SVC(kernel="linear")#LinearSVC()
        #params = {'C':(2,4,8,16,32,64,128,256,512,1024)}
        #clf = GridSearchCV(model_svc, params)
        model_svc.fit(T_train_features, label_train)
        
        # Get Feature importance
        #print("info:",info)
        f_importances(np.abs(model_svc.coef_[0]), ["input"+str(i) for i in range(T_shapelets.shape[0])])
        #print(model_svc.coef_[0]);
        
        after = timeit.time.time()
        predictions = model_svc.predict(T_test_features)
        # Save values
        acc = np.round(1 - (np.where(np.array(predictions)-np.array(label_test) != 0)[0].shape[0]/float(len(predictions))), decimals=5)
        t = np.round(after - before, decimals=2)
        coef = np.array(np.abs(model_svc.coef_[0]))
        coef = coef/np.max(coef)
        info[:,2] = 1-info[:,2]
        data = np.concatenate((info,coef.reshape((coef.shape[0],1))), axis=1)
        print(data)
        print(i,"ran in",t)
        print("Accuracy ( r=",i,"):",acc)
        raw_num.append(i)
        time_arr.append(t)
        accuracy.append(acc)
        if(acc >= 0.99):
            if data_all is None:
                data_all = data
            else:
                data_all = np.concatenate((data_all, data), axis=0)
    return data_all
            
def get_files(path_train, path_test):
    # START Data Read
    
    file_train = open(path_train, 'r')
    file_test = open(path_test, 'r')
    # Create data arrays
    label_train = []
    T_train = []
    label_test = []
    T_test = []
    # TRAIN
    for line in file_train:
        vals = line.replace("\n", "") .split(',')
        label_train.append(vals[0])
        T_train.append(vals[1:])
            # TEST
    for line in file_test:
        vals = line.replace("\n", "") .split(',')
        label_test.append(vals[0])
        T_test.append(vals[1:])
    file_train.close()
    file_test.close()
    del file_train, file_test, line, vals
    # END Data Read
    
    # Get numpy arrays
    label_train = np.array(label_train).astype(np.float)
    T_train = np.array(T_train).astype(np.float)
    
    label_test = np.array(label_test).astype(np.float)
    T_test = np.array(T_test).astype(np.float)
    return T_train, label_train, T_test, label_test

UCR_PATH = "UCRData\\"
UCR_train = [os.path.join(dp, f) for dp, dn, filenames in os.walk(UCR_PATH) for f in filenames if os.path.splitext(f)[1] != '.csv' and "TRAIN" in f]
UCR_test = [os.path.join(dp, f) for dp, dn, filenames in os.walk(UCR_PATH) for f in filenames if os.path.splitext(f)[1] != '.csv' and "TEST" in f]
GUN_train = "Gun_Point/Gun_Point_TRAIN"
GUN_test = "Gun_Point/Gun_Point_TEST"

x_train, y_train, x_test, y_test = get_files(GUN_train, GUN_test)
#data = train_scoring(x_train, y_train, x_test, y_test, seed=2000, dtw=False)

#shapelets.test_approx_accuracy(x_train[0:10],y_train[0:10],1,seed=2000)

def model_scoring(data):
    X_train = data[:-50,:-1]
    Y_train = data[:-50,-1]
    X_test = data[-50:,:-1]
    Y_test = data[-50:,-1]
    
    model = GradientBoostingRegressor()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    RMS_error = np.sqrt(np.mean((Y_test - pred)**2))
    print("RMS Error:",RMS_error)
    print(model.feature_importances_)



