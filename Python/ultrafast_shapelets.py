import numpy as np
import pandas as pd

### Choose p random subsequences for a multivariate time series
def ts_shapelet_features_univariate(T,p):
    n = T.shape[0]
    m = T.shape[1]
    #s = T.shape[2] if len(T.shape) > 2 else 1
    subsequences = []
    total_num_subsequences = float(np.sum(np.array([num_subsequences(T,i) for i in range(3, m+1)])))
    l_vals = np.random.choice(
        np.arange(3,m+1),
        size=p,
        replace=True,
        p=np.array([num_subsequences(T,l)/total_num_subsequences for l in range(3,m+1)]))
    # range over shapelet length
    for l in l_vals:
        i = np.random.randint(0,n)      # random row in data
        j = np.random.randint(0,m-l+1)  # random start time
        subsequences.append([i,j,l])  # add shapelet parameters to subsequences
        # range over Streams (multivariate variables)
        #for k in range(0, s):
        # generate number of shapelets according to possible number of shapelets 
        # for a given subsequence length
    
    # Generate transformed dataset using T and subsequences
    X = np.zeros((n,p))
    for j in range(0, p):
        i_, j_, l = subsequences[j]
        for i in range(0, n):
            X[i,j] = minDist(T[i_, j:j+l], T[i])
    return X, subsequences

def ts_features_from_shapelets(T, shapelets):
    n = T.shape[0]
    p = shapelets.shape[0]
    X = np.zeros((n,p))
    for j in range(0, p):
        i_, j_, l = shapelets[j]
        for i in range(0, n):
            X[i,j] = minDist(T[i_, j:j+l], T[i])
    return X
    
def minDist(S, T):
    l = S.shape[0]
    return np.min(np.array([ np.sqrt(np.sum((S-T[i+l])**2)) for i in range(0, T.shape[0] - l + 1)]))
    #[np.sqrt(np.sum(np.array([ (S[j] - T[i+j])**2   for j in range(0, S.shape[0])] )))

### get the number of subsequences of length i in a dataset T
def num_subsequences(T,i):
    return np.sum(np.array(list(map(len, T))) - (i-1))

