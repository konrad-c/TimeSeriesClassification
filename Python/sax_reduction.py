import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ultrafast_shapelets as shapelets
import pysax
import timeit
import os

# Classification
#from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV

def shapelet_discovery(T_train, label_train, T_test, label_test, shapelet_num, approx=False, useSVC=True):
    # Train model on shapelets generated by TRAIN
    before = timeit.time.time()
    if approx:
        T_train_features, T_shapelets = T_train_features, T_shapelets = shapelets.motif_feature_approx(T_train, label_train, shapelet_num, 200, reduced=True)
    else:
        T_train_features, T_shapelets = shapelets.ts_shapelet_features_univariate(T_train,shapelet_num)
    # Get feature distances from shapelets generated by TRAIN
    T_test_features = shapelets.ts_features_from_shapelets(T_test, T_train, np.array(T_shapelets))
    ### Decide on model:
    if useSVC:
        model_svc = LinearSVC()
        params = {'C':(2,4,8,16,32,64,128,256,512,1024)}
        model = GridSearchCV(model_svc, params)
    else:
        model = XGBClassifier(n_estimators=100, max_depth=5)
        
    model.fit(T_train_features, label_train)
    after = timeit.time.time()
    predictions = model.predict(T_test_features)
    
    print("(p=",shapelet_num,"r=10000) Ran in",after-before)
    print("Accuracy (Test):",1 - (np.where(np.array(predictions)-np.array(label_test) != 0)[0].shape[0]/float(len(predictions))))
    print("Avg 5-fold Cross Validation Accuracy (Test):",np.mean(cross_val_score(model, T_test_features,label_test, cv=5)))

def plot_results(out_filename, T_train, label_train, T_test, label_test, euclidean=False, nRuns=3, seed=-1):
    """
    Run timer and plot time complexity/accuracy
    """
    print("Starting",out_filename, ("Euclidean" if euclidean else "SAX"))
    before_dataset = timeit.time.time()
    if euclidean:
        shapelet_nums = np.arange(1,52,5)#np.arange(10,200,10)
        pruned_nums = [-1]
    else:
        shapelet_nums = np.arange(50,301,50)
        pruned_nums = np.arange(1,52,5)
    raw_num = []
    p_num = []
    time_arr = []
    accuracy = []
    out_file = open(out_filename,"w+")
    out_file.write("raw_shapelet_num,pruned_shapelet_num,time,accuracy\n")
    for j in pruned_nums:
        for i in shapelet_nums:
            acc = 0
            t = 0
            for _ in range(nRuns):
                # Check time
                before = timeit.time.time()
                if euclidean:
                    T_train_features, T_shapelets = shapelets.ts_shapelet_features_univariate(T_train, i, seed=seed)
                else:
                    T_train_features, T_shapelets = shapelets.motif_feature_approx(T_train, label_train, j, i, reduced=True, seed=seed)
                # Check accuracy
                T_test_features = shapelets.ts_features_from_shapelets(T_test, T_train, np.array(T_shapelets))
                # Normalize data
                scaler = StandardScaler().fit(T_train_features)
                T_train_features = scaler.transform(T_train_features)
                T_test_features = scaler.transform(T_test_features)
                # Predict
                model_svc = LinearSVC()
                params = {'C':(2,4,8,16,32,64,128,256,512,1024)}
                clf = GridSearchCV(model_svc, params)
                clf.fit(T_train_features, label_train)
                after = timeit.time.time()
                predictions = clf.predict(T_test_features)
                # Save values
                acc = acc + np.round(1 - (np.where(np.array(predictions)-np.array(label_test) != 0)[0].shape[0]/float(len(predictions))), decimals=5)
                t = t + np.round(after - before, decimals=2)
            t= t/float(nRuns)
            acc = acc/float(nRuns)
            print(j,i,"ran in",t)
            print("Accuracy ( p=",j,"r=",i,"):",acc)
            raw_num.append(i)
            p_num.append(j)
            time_arr.append(t)
            accuracy.append(acc)
            out_file.write(str(i)+","+str(j)+","+str(t)+","+str(acc)+"\n")
    out_file.close()
    after_dataset = timeit.time.time()
    print("Finished",out_filename, ("Euclidean" if euclidean else "SAX"),"in",np.round(before_dataset-after_dataset,decimals=2),"s")

def SAX_NN(T_train, label_train, T_test, label_test, reduced=False):
    # Create SAX encoding model
    sax_model = pysax.SAXModel(alphabet="ABCDEFGHIJKLMNOPQ")
    
    # Create SAX representatiosn of time series'
    sax_rep = np.array([sax_model.symbolize(s) for s in T_train])
    sax_rep_test = np.array([sax_model.symbolize(s) for s in T_test])
    
    if reduced:
        sax_rep = np.array([''.join(ch for ch, _ in itertools.groupby(s)) for s in sax_rep])
        sax_rep_strings = np.array(["".join(s) for s in sax_rep])
        sax_rep_test = np.array([''.join(ch for ch, _ in itertools.groupby(s)) for s in sax_rep_test])
        sax_rep_strings_test = np.array(["".join(s) for s in sax_rep_test])
    else:
        sax_rep_strings = np.array(["".join(s) for s in sax_rep])
        sax_rep_strings_test = np.array(["".join(s) for s in sax_rep_test])
    
    # TEST nearest neighbour on TEST set:
    accuracy = 0
    for i in range(0, len(T_test)):
        word = sax_rep_strings_test[i]
        min_dist_index = np.argmin(np.array([sax_model.symbol_distance(word, w) for w in sax_rep_strings]))
        min_dist = sax_model.symbol_distance(word, sax_rep_strings[min_dist_index])
        classification = label_train[min_dist_index]
        if(classification == label_test[i]):
            accuracy = accuracy + 1
        #print(i,":",word,"matched with",sax_rep_strings[min_dist_index],"dist :",min_dist, " CLASSIFICATION: predicted",classification,"real",label_test[i])
    print("Total Error :",1.0 - (float(accuracy)/float(len(T_test))))
    print("Total Accuracy :",(float(accuracy)/float(len(T_test))))

def test_files(train_filenames, test_filenames):
    for i in range(len(train_filenames)):
        # START Data Read
        path_train = train_filenames[i]
        path_test = test_filenames[i]
        
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
        # Run tests
        plot_results(path_train.split("\\")[-1].replace("TRAIN", "results_euclidean.csv"),
            T_train,
            label_train,
            T_test,
            label_test,
            euclidean=True,
            nRuns=3,
            seed=2082)
        plot_results(path_train.split("\\")[-1].replace("TRAIN", "results_SAX.csv"),
            T_train,
            label_train,
            T_test,
            label_test,
            euclidean=False,
            nRuns=3,
            seed=2082)

UCR_PATH = "UCRData\\"
UCR_train = [os.path.join(dp, f) for dp, dn, filenames in os.walk(UCR_PATH) for f in filenames if os.path.splitext(f)[1] != '.csv' and "TRAIN" in f]
UCR_test = [os.path.join(dp, f) for dp, dn, filenames in os.walk(UCR_PATH) for f in filenames if os.path.splitext(f)[1] != '.csv' and "TEST" in f]
GUN_train = ["Gun_Point/Gun_Point_TRAIN"]
GUN_test = ["Gun_Point/Gun_Point_TEST"]

test_files(UCR_train, UCR_test)
#shapelet_discovery(T_train, label_train, T_test, label_test, 500, approx=False, useSVC=True)
#SAX_NN(T_train, label_train, T_test, label_test, reduced=False)
#SAX_NN(T_train, label_train, T_test, label_test, reduced=True)
