import numpy as np
import pandas as pd
import pysax
import itertools
from fastdtw import fastdtw,dtw

MAX_R =1000
### Choose p random subsequences for a multivariate time series
### Based on pseudocode in Ultra-fast Shapelets
def ts_shapelet_features_univariate(T,p, seed=-1):
    n = T.shape[0]
    m = T.shape[1]
    #s = T.shape[2] if len(T.shape) > 2 else 1
    subsequences = []
    total_num_subsequences = float(np.sum(np.array([num_subsequences(T,i) for i in range(3, m+1)])))
    # range over shapelet length
    
    if seed > 0:
        np.random.seed(seed)
        l_vals = np.random.choice(
            np.arange(3,m+1),
            size=MAX_R,
            replace=True,
            p=np.array([num_subsequences(T,l)/total_num_subsequences for l in range(3,m+1)]))
        l_vals = l_vals[:p]
    else:
        l_vals = np.random.choice(
            np.arange(3,m+1),
            size=p,
            replace=True,
            p=np.array([num_subsequences(T,l)/total_num_subsequences for l in range(3,m+1)]))
            
    # range over shapelet length
    for l_i in range(0, l_vals.shape[0]):
        l = l_vals[l_i]
        i = np.random.randint(0,n)      # random row in data
        j = np.random.randint(0,m-l+1)  # random start time
        subsequences.append([i,j,l])  # add shapelet parameters to subsequences
        # range over Streams (multivariate variables)
        #for k in range(0, s):
        # generate number of shapelets according to possible number of shapelets 
        # for a given subsequence length
    
    # Generate transformed dataset using T and subsequences
    X = np.zeros((n,len(subsequences)))
    for j in range(0, len(subsequences)):
        i_, j_, l = subsequences[j]
        for i in range(0, n):
            X[i,j] = minDist(T[i_, j:j+l], T[i])
            #X[i,j] = minNormDist(T[i_, j:j+l], T[i])
    return X, subsequences

def ts_features_from_shapelets(T, train, shapelets, dtw=True):
    n = T.shape[0]
    p = shapelets.shape[0]
    X = np.zeros((n,p))
    for j in range(0, p):
        i_, j_, l = shapelets[j]
        for i in range(0, n):
            if dtw:
                X[i,j] = minDistDTW(train[i_, j:j+l], T[i])
            else:
                X[i,j] = minDist(train[i_, j:j+l], T[i])
    return X
    
def minDist(S, T):
    l = S.shape[0]
    minD = np.inf
    for i in range(0, T.shape[0] - l):
        t_dist = np.sqrt(np.sum(np.power(S-T[i:i+l], 2)))
        minD = minD if minD < t_dist else t_dist
    return minD
    #return np.min(np.array([ np.sqrt(np.sum((S-T[i:i+l])**2)) for i in range(0, T.shape[0] - l)]))
    #[np.sqrt(np.sum(np.array([ (S[j] - T[i+j])**2   for j in range(0, S.shape[0])] )))

def minDistDTW(S, T):
    minD = np.inf
    for a in range(T.shape[0]-1):
        for b in range(a+1,T.shape[0]):
            t_dist, _ = fastdtw(S, T[a:b+1])
            minD = minD if minD < t_dist else t_dist
    return minD

def minNormDist(S,T):
    l = S.shape[0]
    minD = np.inf
    mu = np.mean(T)
    sigma = np.std(T)
    T = (T-mu)/sigma
    for i in range(0, T.shape[0] - l):
        t_dist = np.sqrt(np.sum(np.power(S-T[i:i+l], 2)))
        minD = minD if minD < t_dist else t_dist
    return minD

def minDistSAX(S,T,sax_model):
    l = len(S)
    minD = np.inf
    for i in range(0, len(T) - l):
        t_dist = sax_model.symbol_distance(S,T[i:i+l])
        minD = minD if minD < t_dist else t_dist
    return minD

### get the number of subsequences of length i in a dataset T
def num_subsequences(T,i):
    return np.sum(np.array(list(map(len, T))) - (i-1))

### Generate p pruned shapelets and generate r random shapelets for pruning
def motif_feature_approx(T,Y,p,r, reduced=False, seed=-1, dtw=True):
    ########### Convert T to SAX approx
    # Create General SAX encoding model with alphabet of size 13
    sax_model = pysax.SAXModel(alphabet="ABCDEFGHIJKLM")
    # Create SAX representatiosn of time series'
    T_sax = np.array([sax_model.symbolize(s) for s in T])
    # Create string representations
    T_sax = np.array(["".join(s) for s in T_sax])
    
    ######### Generate Random shapelets checking with SAX distance #########
    n = T.shape[0]
    m = T.shape[1]
    subsequences_raw = np.zeros((r,3), dtype=np.int)
    total_num_subsequences = float(np.sum(np.array([num_subsequences(T,i) for i in range(3, m+1)])))
    if seed > 0:
        np.random.seed(seed)
        l_vals = np.random.choice(
            np.arange(3,m+1),
            size=MAX_R,
            replace=True,
            p=np.array([num_subsequences(T,l)/total_num_subsequences for l in range(3,m+1)]))
        l_vals = l_vals[:r]
    else:
        l_vals = np.random.choice(
            np.arange(3,m+1),
            size=r,
            replace=True,
            p=np.array([num_subsequences(T,l)/total_num_subsequences for l in range(3,m+1)]))
    # range over shapelet length
    for l_i in range(0, l_vals.shape[0]):
        l = l_vals[l_i]
        i = np.random.randint(0,n)      # random row in data
        j = np.random.randint(0,m-l+1)  # random start time
        subsequences_raw[l_i] = np.array([i,j,l])  # add shapelet parameters to subsequences
    subsequences_raw = np.array(subsequences_raw)
    backup_subsequences = subsequences_raw
    
    ######## Prune Raw Shapelets ########
    if reduced:
        subseq_vals = np.array([''.join(ch for ch, _ in itertools.groupby(T_sax[i][j:j+l])) for i,j,l in subsequences_raw])
        t_len = np.vectorize(len)
        T_sax = np.array([''.join(ch for ch, _ in itertools.groupby(s)) for s in T_sax])
        mask = np.where((t_len(subseq_vals) > 1) * (t_len(subseq_vals) <= np.min(t_len(T_sax)))) 
        subseq_vals = subseq_vals[mask]
        subsequences_raw = subsequences_raw[mask]
        _, unique_mask = np.unique(subseq_vals, return_index=True)
        subseq_vals = subseq_vals[unique_mask]
        subsequences_raw = subsequences_raw[unique_mask]
    else:
        subseq_vals = np.array([T_sax[i][j:j+l] for i,j,l in subsequences_raw])
    
    d_matrix = np.zeros((subseq_vals.shape[0], pd.unique(Y).shape[0]))
    SAX_vec = np.vectorize(minDistSAX)
    for t in range(0, pd.unique(Y).shape[0]):
        target = pd.unique(Y)[t]
        T_sax_target = T_sax[np.where(Y == target)]
        for s in range(0, len(subseq_vals)):
            #print(SAX_vec(T_sax_target, subseq_vals[s], sax_model))
            d_matrix[s,t] = np.sum(np.array([minDistSAX(subseq_vals[s], w, sax_model) for w in T_sax_target])) #SAX_vec(T_sax_target, subseq_vals[s], sax_model))#
    # remove zero and inf distances:
    zeroinf_mask = ~((np.sum(d_matrix,axis=1) == 0) + (np.sum(d_matrix,axis=1) == np.inf))
    d_matrix = d_matrix[zeroinf_mask]
    subseq_vals = subseq_vals[zeroinf_mask]
    subsequences_raw = subsequences_raw[zeroinf_mask]
    # Calculate importance of shapelets
    # p(x_i) = [(max(x) - x_i)+1]/sum([(max(x) - x_i)+1])
    inv_d_matrix = -d_matrix + np.amax(d_matrix,axis=1,keepdims=True) + np.amin(d_matrix, axis=1, keepdims=True)
    prob_matrix = (inv_d_matrix+1)/np.sum((inv_d_matrix+1),axis=1,keepdims=True)
    # This maximises distance: prob_matrix = (d_matrix+1)/np.sum((d_matrix+1),axis=1,keepdims=True)
    entropy = -np.sum(prob_matrix*np.log2(prob_matrix),axis=1)
    #gap = np.amax(d_matrix, axis=1) - np.amin(d_matrix, axis=1)
    #gap = 1 - ((gap - np.min(gap))/np.max(gap))
    quality = entropy#(entropy+gap)/2
    #print(np.concatenate((d_matrix, quality.reshape((d_matrix.shape[0],1))), axis=1))
    #gap = np.abs(d_matrix[:,0] - d_matrix[:,1])
    p = min(p, subseq_vals.shape[0])
    #print(np.concatenate((d_matrix, quality.reshape((d_matrix.shape[0],1))), axis=1)[np.argpartition(quality, p-1)[:p]])
    subsequences_pruned = subsequences_raw[np.argpartition(quality, p-1)[:p]]
    ## Check subsequences isn't empty
    if subsequences_pruned.shape[0] == 0:
        subsequences_pruned = backup_subsequences[np.random.choice(backup_subsequences.shape[0],size=p, replace=False)]
    # Generate transformed dataset using T and subsequences
    X = np.zeros((n,p))
    for j in range(0, p):
        i_, j_, l = subsequences_pruned[j]
        for i in range(0, n):
            if dtw:
                X[i,j] = minDistDTW(train[i_, j:j+l], T[i])
            else:
                X[i,j] = minDist(T[i_, j:j+l], T[i])
    return X, subsequences_pruned, np.concatenate((d_matrix, quality.reshape((d_matrix.shape[0],1))), axis=1)[np.argpartition(quality, p-1)[:p]]


def test_approx_accuracy(T,Y,r,seed=-1):
    ########### Convert T to SAX approx
    # Create General SAX encoding model with alphabet of size 13
    sax_model = pysax.SAXModel(alphabet="ABCDEFGHIJKLM")
    # Create SAX representatiosn of time series'
    T_sax = np.array([sax_model.symbolize(s) for s in T])
    # Create string representations
    T_sax = np.array(["".join(s) for s in T_sax])
    ######### Generate Random shapelets checking with SAX distance #########
    n = T.shape[0]
    m = T.shape[1]
    subsequences_raw = np.zeros((r,3), dtype=np.int)
    total_num_subsequences = float(np.sum(np.array([num_subsequences(T,i) for i in range(3, m+1)])))
    if seed > 0:
        np.random.seed(seed)
        l_vals = np.random.choice(
            np.arange(3,m+1),
            size=MAX_R,
            replace=True,
            p=np.array([num_subsequences(T,l)/total_num_subsequences for l in range(3,m+1)]))
        l_vals = l_vals[:r]
    else:
        l_vals = np.random.choice(
            np.arange(3,m+1),
            size=r,
            replace=True,
            p=np.array([num_subsequences(T,l)/total_num_subsequences for l in range(3,m+1)]))
    # range over shapelet length
    for l_i in range(0, l_vals.shape[0]):
        l = l_vals[l_i]
        i = np.random.randint(0,n)      # random row in data
        j = np.random.randint(0,m-l+1)  # random start time
        subsequences_raw[l_i] = np.array([i,j,l])  # add shapelet parameters to subsequences
    subsequences_raw = np.array(subsequences_raw)
    backup_subsequences = subsequences_raw
    
    ######## Prune Raw Shapelets ########
    subseq_vals = np.array([''.join(ch for ch, _ in itertools.groupby(T_sax[i][j:j+l])) for i,j,l in subsequences_raw])
    t_len = np.vectorize(len)
    T_sax = np.array([''.join(ch for ch, _ in itertools.groupby(s)) for s in T_sax])
    mask = np.where((t_len(subseq_vals) > 1) * (t_len(subseq_vals) <= np.min(t_len(T_sax)))) 
    subseq_vals = subseq_vals[mask]
    subsequences_raw = subsequences_raw[mask]
    _, unique_mask = np.unique(subseq_vals, return_index=True)
    subseq_vals = subseq_vals[unique_mask]
    subsequences_raw = subsequences_raw[unique_mask]
    
    d_matrix_SAX = np.zeros((subseq_vals.shape[0], n))
    d_matrix_DTW = np.zeros((subseq_vals.shape[0], n))
    SAX_vec = np.vectorize(minDistSAX)
    DTW_vec = np.vectorize(minDistDTW)
    for t in range(0, n):
        w_sax = T_sax[t]
        w_dtw = T[t]
        for s in range(0, len(subseq_vals)):
            i_, j_, l = subsequences_raw[s]
            d_matrix_SAX[s,t] = minDistSAX(subseq_vals[s], w_sax, sax_model)
            d_matrix_DTW[s,t] = minDistDTW(T[i_, j:j+l], w_dtw)
    d_matrix_SAX = d_matrix_SAX.flatten()
    d_matrix_DTW = d_matrix_DTW.flatten()
    error = np.abs(d_matrix_SAX - d_matrix_DTW)
    return d_matrix_SAX, d_matrix_DTW, error

