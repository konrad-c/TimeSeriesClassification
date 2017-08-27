import os
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = "NormalizationData\\OriginalDS-all\\"
DATA_PATH_OUT = "NormalizationData\\GesturesRaw\\"
gestures_data = [os.path.join(dp, f) for dp, dn, filenames in os.walk(DATA_PATH) for f in filenames]

classification = np.zeros(3811, dtype='<U5')
time = np.zeros((3811,889))
x = np.zeros((3811,889))
y = np.zeros((3811,889))
z = np.zeros((3811,889))
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

del vals,f,t_time,t_class,t_x,t_y,t_z,t_len,i

# plot the times
for i in range(len(gestures_data)):
    plt.plot(time[i], y[i])
plt.show()

# plot the times
for i in range(len(gestures_data)):
    plt.plot(y[i])
plt.show()

### Write data to files ###
out_x = open(DATA_PATH_OUT + "GesturesX.csv", "w+")
out_y = open(DATA_PATH_OUT + "GesturesY.csv", "w+")
out_z = open(DATA_PATH_OUT + "GesturesZ.csv", "w+")
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

