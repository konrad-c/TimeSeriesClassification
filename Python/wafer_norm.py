import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_max_length(data):
    lengths = []
    for i in range(len(data)):
        filename = data[i]
        f = open(filename)
        t_len = 0
        for line in f:
            t_len = t_len + 1
        f.close()
        lengths.append(t_len)
    return max(lengths)
    
def get_lengths(data):
    lengths = []
    for i in range(len(data)):
        filename = data[i]
        f = open(filename)
        t_len = 0
        for line in f:
            t_len = t_len + 1
        f.close()
        lengths.append(t_len)
    return np.array(lengths)

def raw_wafer(filenames_normal, filenames_abnormal, stretch_length=None, rolling_average=False):
    rows = len(filenames_normal) + len(filenames_abnormal)
    max_len = max(get_max_length(filenames_normal), get_max_length(filenames_abnormal))
    classification = np.zeros(rows, dtype='<U10')
    
    values_normal = np.zeros((len(filenames_normal),max_len))
    values_abnormal = np.zeros((len(filenames_abnormal),max_len))
    
    for i in range(len(filenames_normal)):
        filename = filenames_normal[i]
        f = open(filename)
        for j,line in enumerate(f):
            vals = line.split("\t")
            if len(vals) > 1:
                values_normal[i,j] = float(vals[-2])
            else:
                print("normal",i)
        f.close()
    for i in range(len(filenames_abnormal)):
        filename = filenames_abnormal[i]
        f = open(filename)
        for j,line in enumerate(f):
            vals = line.split("\t")
            if len(vals) > 1:
                values_abnormal[i,j] = float(vals[-2])
            else:
                print("abnormal",i)
        f.close()
    values = np.concatenate((values_normal,values_abnormal), axis=0)
    if stretch_length is not None:
        lengths = np.concatenate((get_lengths(filenames_normal), get_lengths(filenames_abnormal)))
        values = stretch_values(values,lengths,stretch_length,rolling_average)
    classification[:len(filenames_normal)] = "normal"
    classification[len(filenames_normal):] = "abnormal"
    
    return classification, values
    
def stretch_values(values,lengths,s_length, rolling_average=False):
    out_vals = np.zeros((values.shape[0], s_length),dtype=np.float)
    for i in range(values.shape[0]):
        t_len = lengths[i]
        for t in range(s_length):
            prop = float(t)/float(s_length)
            out_vals[i,t] = values[i,int(t_len*prop)]
            if rolling_average:
                window = int(s_length/t_len)
                if window - 1 > 0:
                    half_w = max(int(window/2), 1)
                    for t in range(s_length):
                        val_copy = values[i]
                        out_vals[i,t] = np.mean(val_copy[max(0,t-half_w):min(s_length,t+half_w+1)])
    return out_vals

def out_data(classification,x,path_out):
    ### Write data to files ###
    out_x = open(path_out + "Wafer.csv", "w+")
    classification = pd.factorize(classification)[0]
    for i in range(x.shape[0]):
        str_x = str(classification[i]) + "," + np.array2string(x[i], separator=",").replace("\n", "").replace(" ", "").replace("[", " ").replace("]", "").replace(".,", ".0,") + "\n"
        out_x.write(str_x)
    out_x.close()
    
def plot_sample(x, sample_size=4):
    sample_x = x[np.random.choice(x.shape[0],size=sample_size,replace=True)]
    for i in sample_x:
        plt.plot(i)
    plt.show()

#def main(): 
#    DATA_PATH = "NormalizationData\\waferRaw\\"
#    wafer_normal = [os.path.join(dp, f) for dp, dn, filenames in os.walk(DATA_PATH + "normal\\") for f in filenames]
#    wafer_abnormal = [os.path.join(dp, f) for dp, dn, filenames in os.walk(DATA_PATH + "abnormal\\") for f in filenames]
#    DATA_PATH_OUT = "NormalizationData\\WaferClean\\"
    #classification,x = raw_wafer(wafer_normal,wafer_abnormal, stretch_length=None, rolling_average=False)
    #plot_sample(x,4)
    #out_data(classification,x,DATA_PATH_OUT + "Truncated")
    
#    classification,x = raw_wafer(wafer_normal,wafer_abnormal, stretch_length=152, rolling_average=False)
#    data = np.array(pd.read_csv("NormalizationData/WaferClean/UCRWafer.csv"))[:,1:]
#    plot_sample(x,20)
#    plot_sample(data,20)
#    out_data(classification,x,DATA_PATH_OUT + "Stretched")
    
#    classification,x = raw_wafer(wafer_normal,wafer_abnormal, stretch_length=152, rolling_average=True)
#    plot_sample(x,4)
#    out_data(classification,x,DATA_PATH_OUT + "StretchedAvg")

#main()



