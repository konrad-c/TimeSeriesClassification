import numpy as np
import pandas as pd
import pysax
import itertools

### Choose p random subsequences for a multivariate time series
### Based on pseudocode in Ultra-fast Shapelets
def ts_shapelet_features_univariate(T,p):
    n = T.shape[0]
    m = T.shape[1]
    #s = T.shape[2] if len(T.shape) > 2 else 1
    subsequences = []
    total_num_subsequences = float(np.sum(np.array([num_subsequences(T,i) for i in range(3, m+1)])))
    f = p/total_num_subsequences
    #l_vals = np.random.choice(
    #    np.arange(3,m+1),
    #    size=p,
    #    replace=True,
    #    p=np.array([num_subsequences(T,l)/total_num_subsequences for l in range(3,m+1)]))
    # range over shapelet length
    for l in range(3, m+1):#l_vals:
        for _ in range(int(np.round(f*num_subsequences(T,l)))):
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

def ts_features_from_shapelets(T, train, shapelets):
    n = T.shape[0]
    p = shapelets.shape[0]
    X = np.zeros((n,p))
    for j in range(0, p):
        i_, j_, l = shapelets[j]
        for i in range(0, n):
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
def motif_feature_approx(T,Y,p,r, reduced=False):
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
    
    ######## Prune Raw Shapelets ########
    d_matrix = np.zeros((r, pd.unique(Y).shape[0]))
    if reduced:
        subseq_vals = [''.join(ch for ch, _ in itertools.groupby(T_sax[i][j:j+l])) for i,j,l in subsequences_raw]
        T_sax = np.array([''.join(ch for ch, _ in itertools.groupby(s)) for s in T_sax])
    else:
        subseq_vals = [T_sax[i][j:j+l] for i,j,l in subsequences_raw]
        
    for t in range(0, pd.unique(Y).shape[0]):
        target = pd.unique(Y)[t]
        if reduced:
            T_sax_target = np.array([''.join(ch for ch, _ in itertools.groupby(s)) for s in T_sax[np.where(Y == target)] ])
        else:
            T_sax_target = np.array(T_sax[np.where(Y == target)])
        for s in range(0, len(subseq_vals)):
            #subseq = subsequences_raw[s]
            #i,j,l = subseq
            rmotif = subseq_vals[s]#T_sax[i][j:j+l]
            d_matrix[s,t] = np.sum(np.array([minDistSAX(rmotif, w, sax_model) for w in T_sax_target]))
    gap = np.abs(d_matrix[:,0] - d_matrix[:,1])
    subsequences_pruned = subsequences_raw[np.argpartition(-gap, p)[:p]]
    # Generate transformed dataset using T and subsequences
    X = np.zeros((n,p))
    for j in range(0, p):
        i_, j_, l = subsequences_pruned[j]
        for i in range(0, n):
            X[i,j] = minDist(T[i_, j:j+l], T[i])
    return X, subsequences_pruned


