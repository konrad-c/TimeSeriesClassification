import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ultrafast_shapelets as shapelets
#import pysax

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# START Data Read
gun_path_train = "UCRData/Gun_Point/Gun_Point_TRAIN"
gun_file_train = open(gun_path_train, 'r')
gun_path_test = "UCRData/Gun_Point/Gun_Point_TEST"
gun_file_test = open(gun_path_test, 'r')

# Create data arrays
label_train = []
T_train = []
label_test = []
T_test = []
# TRAIN
for line in gun_file_train:
    vals = line.replace("\n", "") .split(',')
    label_train.append(vals[0])
    T_train.append(vals[1:])
# TEST
for line in gun_file_test:
    vals = line.replace("\n", "") .split(',')
    label_test.append(vals[0])
    T_test.append(vals[1:])
gun_file_train.close()
gun_file_test.close()
del gun_path_train, gun_file_train, gun_path_test, gun_file_test, line, vals
# END Data Read

# Get numpy arrays
label_train = np.array(label_train).astype(np.float)
T_train = np.array(T_train).astype(np.float)

label_test = np.array(label_test).astype(np.float)
T_test = np.array(T_test).astype(np.float)

# Train model on shapelets generated by TRAIN
T_train_features, T_shapelets = shapelets.ts_shapelet_features_univariate(T_train,100)
model = RandomForestClassifier(n_estimators=50)
model.fit(T_train_features, label_train)

# Get feature distances from shapelets generated by TRAIN
T_test_features = ts_features_from_shapelets(T_test, np.array(T_shapelets))
predictions = model.predict(T_test_features)
print("Error:",(predictions-label_test != 0)/float(len(predictions))))


cross_val_score(model, T_shapelets,label - 1, cv=10)




# Create SAX encoding model
sax_model = pysax.SAXModel(alphabet="ABCDEFGHIJKLMNOPQ")

# Create SAX representatiosn of time series'
sax_rep = np.array([sax_model.symbolize(s) for s in series])
sax_reduced = np.array([pd.unique(s) for s in sax_rep])

# Create string representations
sax_rep_strings = np.array(["".join(s) for s in sax_rep])
sax_reduced_strings = np.array(["".join(s) for s in sax_reduced])

for i in range(0,6):
    print(sax_rep_strings[i], "\n", sax_reduced_strings[i], "\t", label[i])
print("Mean Reduced length:",np.mean(np.array(list(map(len, sax_reduced_strings)))))
np.where(np.array(list(map(len, sax_reduced_strings))) != round(np.mean(np.array(list(map(len, sax_reduced_strings))))))

# TEST take one out nearest neighbour:
def hold_one():
    accuracy = 0
    for i in range(0, len(series)):
        word = sax_rep_strings[i]
        data = np.delete(sax_rep_strings,i)
        min_dist_index = np.argmin(np.array([sax_model.symbol_distance(word, w) for w in data]))
        min_dist = sax_model.symbol_distance(word, data[min_dist_index])
        classification = np.delete(label,i)[min_dist_index]
        if(classification == label[i]):
            accuracy = accuracy + 1
        print(i,":",word,"matched with",data[min_dist_index],"dist :",min_dist, " CLASSIFICATION: predicted",classification,"real",label[i])
    print("Total Error :",1.0 - (float(accuracy)/float(len(series))))
    
    
def find_shapelets(k):
    max_shapelet_length = np.min(np.array(list(map(len, series))))
    
    
    
hold_one()


