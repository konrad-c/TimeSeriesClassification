import os
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = "NormalizationData\\CricketDataRaw\\"
DATA_PATH_OUT = "NormalizationData\\CricketClean\\"
cricket_data = [os.path.join(dp, f) for dp, dn, filenames in os.walk(DATA_PATH) for f in filenames]

def get_max_length(cricket_data):
    lengths = []
    for i in range(len(cricket_data)):
        filename = cricket_data[i]
        f = open(filename)
        t_len = 0
        for line in f:
            t_len = t_len + 1
        f.close()
        lengths.append(t_len)
    return max(lengths)

def clean_data_truncate(cricket_data):
    classification = np.zeros(len(cricket_data), dtype='<U10')
    max_len = get_max_length(cricket_data)
    x = np.zeros((len(cricket_data),max_len))
    y = np.zeros((len(cricket_data),max_len))
    z = np.zeros((len(cricket_data),max_len))
    for i in range(len(cricket_data)):
        filename = cricket_data[i]
        t_class = filename.split("\\")[-1].split("-")[0]
        t_x = []
        t_y = []
        t_z = []
        f = open(filename)
        for line in f:
            vals = line.split(" ")
            t_x.append(vals[0])
            t_y.append(vals[1])
            t_z.append(vals[2])
        f.close()
        t_len = len(t_x)
        classification[i] = t_class
        x[i, 0:t_len] = np.array(t_x)
        x[i, t_len:] = np.ones(x.shape[1] - t_len, dtype=float)*np.array(t_x).astype(float)[-1]
        y[i, 0:t_len] = np.array(t_y)
        y[i, t_len:] = np.ones(y.shape[1] - t_len, dtype=float)*np.array(t_y).astype(float)[-1]
        z[i, 0:t_len] = np.array(t_z)
        z[i, t_len:] = np.ones(z.shape[1] - t_len, dtype=float)*np.array(t_z).astype(float)[-1]
    return classification,x,y,z

def clean_data_stretched(gestures_data, size=315, rolling_average=True):
    classification = np.zeros(len(cricket_data), dtype='<U10')
    max_len = get_max_length(cricket_data)
    x = np.zeros((len(cricket_data),size))
    y = np.zeros((len(cricket_data),size))
    z = np.zeros((len(cricket_data),size))
    for i in range(len(cricket_data)):
        filename = cricket_data[i]
        t_class = filename.split("\\")[-1].split("-")[0]
        t_x = []
        t_y = []
        t_z = []
        f = open(filename)
        for line in f:
            vals = line.split(" ")
            t_x.append(vals[0])
            t_y.append(vals[1])
            t_z.append(vals[2])
        f.close()
        t_len = len(t_x)
        classification[i] = t_class
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
    return classification,x,y,z

def out_data(classification,x,y,z,path_out):
    ### Write data to files ###
    out_x = open(path_out + "CricketX.csv", "w+")
    out_y = open(path_out + "CricketY.csv", "w+")
    out_z = open(path_out + "CricketZ.csv", "w+")
    classification = pd.factorize(classification)[0]
    for i in range(len(cricket_data)):
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
    classification,x,y,z = clean_data_stretched(cricket_data, size=300, rolling_average=True)
    out_data(classification,x,y,z,DATA_PATH_OUT + "StretchedAvg")
    classification,x,y,z = clean_data_stretched(cricket_data, size=300, rolling_average=False)
    out_data(classification,x,y,z,DATA_PATH_OUT + "Stretched")
    ### Truncated
    classification,x,y,z = clean_data_truncate(cricket_data)
    out_data(classification,x,y,z,DATA_PATH_OUT + "Truncated")

main()

#classification,x,y,z = clean_data(gestures_data)
ucr_data = np.array(pd.read_csv("UCRData_long/Cricket_X/Cricket_X_TRAIN"))[:,1:]

#classification,x,y,z = clean_data_timenormal(gestures_data, size=315)
out_data(classification,x,y,z,DATA_PATH_OUT + "Stretched")

# plot the times
sample_size = 10
ucr_sample = ucr_data[np.random.choice(ucr_data.shape[0], size=sample_size, replace=False)]
for i in range(ucr_sample.shape[0]):
    plt.plot(ucr_sample[i])
plt.show()

# plot the times
for i in range(140,150):
    plt.plot(x[i])
plt.show()




