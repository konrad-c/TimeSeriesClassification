import os
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = "NormalizationData\\OriginalDS-all\\"
DATA_PATH_OUT = "NormalizationData\\GesturesRaw\\"
gestures_data = [os.path.join(dp, f) for dp, dn, filenames in os.walk(DATA_PATH) for f in filenames]

def clean_data_truncate(gestures_data):
    classification = np.zeros(len(gestures_data), dtype='<U5')
    time = np.zeros((len(gestures_data),889))
    x = np.zeros((len(gestures_data),889))
    y = np.zeros((len(gestures_data),889))
    z = np.zeros((len(gestures_data),889))
    for i in range(len(gestures_data)):
        filename = gestures_data[i]
        t_class = filename.split("_")[-2]
        t_time = []
        t_x = []
        t_y = []
        t_z = []
        f = open(filename)
        for line in f:
            vals = line.split(",")
            t_time.append(vals[0])
            t_x.append(vals[1])
            t_y.append(vals[2])
            t_z.append(vals[3])
        f.close()
        t_len = len(t_x)
        classification[i] = t_class
        time[i, 0:t_len] = np.array(t_time)
        x[i, 0:t_len] = np.array(t_x)
        x[t_len:] = np.ones(x.shape[1] - t_len, dtype=float)*np.array(t_x).astype(float)[-1]
        y[i, 0:t_len] = np.array(t_y)
        y[t_len:] = np.ones(y.shape[1] - t_len, dtype=float)*np.array(t_y).astype(float)[-1]
        z[i, 0:t_len] = np.array(t_z)
        z[t_len:] = np.ones(z.shape[1] - t_len, dtype=float)*np.array(t_z).astype(float)[-1]
    return classification,x,y,z

def clean_data_stretched(gestures_data, size=315, rolling_average=True):
    classification = np.zeros(len(gestures_data), dtype='<U5')
    time = np.zeros((len(gestures_data),size))
    x = np.zeros((len(gestures_data),size))
    y = np.zeros((len(gestures_data),size))
    z = np.zeros((len(gestures_data),size))
    for i in range(len(gestures_data)):
        filename = gestures_data[i]
        t_class = filename.split("_")[-2]
        t_time = []
        t_x = []
        t_y = []
        t_z = []
        f = open(filename)
        for line in f:
            vals = line.split(",")
            t_time.append(float(vals[0]))
            t_x.append(float(vals[1]))
            t_y.append(float(vals[2]))
            t_z.append(float(vals[3]))
        f.close()
        t_len = len(t_x)
        for t in range(size):
            prop = float(t)/float(size)
            x[i,t] = t_x[int(t_len*prop)]
            y[i,t] = t_y[int(t_len*prop)]
            z[i,t] = t_z[int(t_len*prop)]
        if rolling_average:
            window = int(size/t_len)
            if window - 1 > 0:
                half_w = max(int(window/2), 1)
                for t in range(size):
                    x_copy = x[i]
                    y_copy = y[i]
                    z_copy = z[i]
                    #### Make sure to copy before smoothing
                    x[i,t] = np.mean(x_copy[max(0,t-half_w):min(size,t+half_w+1)])
                    y[i,t] = np.mean(y_copy[max(0,t-half_w):min(size,t+half_w+1)])
                    z[i,t] = np.mean(z_copy[max(0,t-half_w):min(size,t+half_w+1)])
        classification[i] = t_class
    return classification,x,y,z
    
    
def clean_data_timenormal(gestures_data, size=315, rolling_average=True):
    classification = np.zeros(len(gestures_data), dtype='<U5')
    time = np.zeros((len(gestures_data),size))
    x = np.zeros((len(gestures_data),size))
    y = np.zeros((len(gestures_data),size))
    z = np.zeros((len(gestures_data),size))
    for i in range(len(gestures_data)):
        filename = gestures_data[i]
        t_class = filename.split("_")[-2]
        t_time = []
        t_x = []
        t_y = []
        t_z = []
        f = open(filename)
        for line in f:
            vals = line.split(",")
            t_time.append(float(vals[0]))
            t_x.append(float(vals[1]))
            t_y.append(float(vals[2]))
            t_z.append(float(vals[3]))
        f.close()
        t_len = len(t_x)
        t_max = max(t_time)
        for j,t in enumerate(t_time[:-2]): #### Try using regression for intermediary points
            prop = float(t)/float(t_max)
            next_prop = float(t_time[j+1])/float(t_max)
            start = int(size*prop)
            stop = int(size*next_prop)
            x[i,start:stop] = t_x[j]# + (np.arange(stop-start)/float(stop-start))*(t_x[j+1]-t_x[j])
            y[i,start:stop] = t_y[j]# + (np.arange(stop-start)/float(stop-start))*(t_y[j+1]-t_y[j])
            z[i,start:stop] = t_z[j]# + (np.arange(stop-start)/float(stop-start))*(t_z[j+1]-t_z[j])
        x[i,-1] = t_x[-1]
        y[i,-1] = t_y[-1]
        z[i,-1] = t_z[-1]
        if rolling_average:
            window = int(size/t_len)
            if window - 1 > 0:
                half_w = max(int(window/2), 1)
                for t in range(size):
                    x_copy = x[i]
                    y_copy = y[i]
                    z_copy = z[i]
                    #### Make sure to copy before smoothing
                    x[i,t] = np.mean(x_copy[max(0,t-half_w):min(size,t+half_w+1)])
                    y[i,t] = np.mean(y_copy[max(0,t-half_w):min(size,t+half_w+1)])
                    z[i,t] = np.mean(z_copy[max(0,t-half_w):min(size,t+half_w+1)])
        classification[i] = t_class
    return classification,x,y,z

def out_data(classification,x,y,z,path_out):
    ### Write data to files ###
    out_x = open(path_out + "GesturesX.csv", "w+")
    out_y = open(path_out + "GesturesY.csv", "w+")
    out_z = open(path_out + "GesturesZ.csv", "w+")
    classification = pd.factorize(classification)[0]
    for i in range(len(gestures_data)):
        str_x = str(classification[i]) + "," + np.array2string(x[i], separator=",").replace("\n", "").replace(" ", "").replace("[", " ").replace("]", "").replace(".,", ".0,") + "\n"
        str_y = str(classification[i]) + "," + np.array2string(y[i], separator=",").replace("\n", "").replace(" ", "").replace("[", " ").replace("]", "").replace(".,", ".0,") + "\n"
        str_z = str(classification[i]) + "," + np.array2string(z[i], separator=",").replace("\n", "").replace(" ", "").replace("[", " ").replace("]", "").replace(".,", ".0,") + "\n"
        out_x.write(str_x)
        out_y.write(str_y)
        out_z.write(str_z)
    out_x.close()
    out_y.close()
    out_z.close()
    
def main():  ### Stretched
    classification,x,y,z = clean_data_stretched(gestures_data, size=315, rolling_average=True)
    out_data(classification,x,y,z,DATA_PATH_OUT + "StretchedAvg")
    classification,x,y,z = clean_data_stretched(gestures_data, size=315, rolling_average=False)
    out_data(classification,x,y,z,DATA_PATH_OUT + "Stretched")
    ### Time normal
    classification,x,y,z = clean_data_timenormal(gestures_data, size=315, rolling_average=True)
    out_data(classification,x,y,z,DATA_PATH_OUT + "TimeNormalAvg")
    classification,x,y,z = clean_data_timenormal(gestures_data, size=315, rolling_average=False)
    out_data(classification,x,y,z,DATA_PATH_OUT + "TimeNormal")

main()    

#classification,x,y,z = clean_data(gestures_data)
classification,x,y,z = clean_data_stretched(gestures_data, size=315, rolling_average=True)
#classification,x,y,z = clean_data_timenormal(gestures_data, size=315)
out_data(classification,x,y,z,DATA_PATH_OUT + "Stretched")

# plot the times
for i in range(len(gestures_data)):
    plt.plot(time[i], y[i])
plt.show()

# plot the times
for i in range(6,8):
    plt.plot(x[i])
plt.show()





