import os
import matplotlib.pyplot as plt

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
data = pd.read_csv("Gun_Point/Gun_Point_TRAIN", header=None)
data_test = pd.read_csv("Gun_Point/Gun_Point_TEST", header=None)
data = data * 10
scaler.fit(data)
data_scaled  = scaler.transform(data)
data_test_scaled = scaler.transform(data_test)

plt.plot(data_scaled)

DATA_PATH = "DSTEmporalOriginal\\"
gestures_data = [os.path.join(dp, f) for dp, dn, filenames in os.walk(DATA_PATH) for f in filenames]

classification = []
x = []
y = []
z = []
for filename in gestures_data:
    classification = []
    t_x = []
    t_y = []
    t_z = []
    f = open(filename)
    for line in f:
        vals = line.split(",")
        t_x.append(vals[1])
        t_y.append(vals[2])
        t_z.append(vals[3])
    x.append(t_x)
    y.append(t_y)
    z.append(t_z)
    f.close()

