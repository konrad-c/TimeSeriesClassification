import cython
import multiprocessing as mp
from cython.parallel import prange
cimport openmp
from libc.math cimport sqrt
from numpy.math cimport INFINITY
import numpy as np
cimport numpy as np
DINT = np.int
DFLOAT = np.float
ctypedef np.int_t DINT_t
ctypedef np.float_t DFLOAT_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef float DTWDistance(double[:] x, double[:] y, double w, double[:,:] DTW) nogil:
    cdef int N = x.shape[0]
    cdef int M = y.shape[0]
    cdef int i, j
    cdef double tmp
    #cdef double[:,:] DTW = np.empty((N, M), dtype=float)
    for i in range(1, N):
        for j in range(1, M):
            tmp = x[i] - y[j]
            DTW[i][j] = tmp*tmp + fmin(DTW[i-1][j], 
                                       DTW[i-1][j-1],
                                       DTW[i][j-1])
    return DTW[N-1][M-1]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef fmin(double a, double b, double c):
    if b < a:
        a = b
    if c < a:
        a = c
    return a

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef float seqmin(double[:] s, int begin, int end) nogil:
    cdef int i
    cdef double minval
    minval = INFINITY
    for i in range(begin,end):
        if s[i] < minval:
            minval = s[i]
    return minval

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)  
cdef float seqmax(double[:] s, int begin, int end) nogil:
    cdef int i
    cdef double maxval
    maxval = -INFINITY
    for i in range(begin,end):
        if s[i] > maxval:
            maxval = s[i]
    return maxval

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline int cmax(int a, int b) nogil: return a if a >= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline int cmin(int a, int b) nogil: return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef float LB_Keogh(double[:] s1, double[:] s2, int r) nogil:
    cdef double lower_bound, upper_bound, temp, val
    cdef int i 
    cdef int N = s1.shape[0]
    cdef double LB_sum = 0
    for i in range(0, N-r):
        val = s1[i]
        lower_bound=seqmin(s2,cmax(i-r, 0),i+r)
        upper_bound=seqmax(s2,cmax(i-r, 0),i+r)
        if i>upper_bound:
            temp = val-upper_bound
            LB_sum=LB_sum+temp*temp
        elif i<lower_bound:
            temp = val-lower_bound
            LB_sum=LB_sum+temp*temp
    return sqrt(LB_sum)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int[:] CNN_predict(double[:,:] x_train, int[:] y_train, double[:,:] x_test, int w):
    cdef int[:] predictions = np.empty(x_test.shape[0], dtype=int)
    cdef int i, j, closest_seq
    cdef int N = x_train.shape[0]
    cdef int M = x_test.shape[0]
    cdef double min_dist,dist
    cdef double[:,:,:] DTW = np.empty((mp.cpu_count(), x_test.shape[1], x_test.shape[1]), dtype=DFLOAT)
    cdef int thread_number
    for i in prange(M, nogil=True):
        thread_number = openmp.omp_get_thread_num()
        min_dist = INFINITY
        for j in range(N):
            if LB_Keogh(x_test[i], x_train[j], 5) < min_dist:
                dist = DTWDistance(x_test[i], x_train[j], w, DTW[thread_number])
                if dist < min_dist:
                    min_dist = dist
                    closest_seq = j
        predictions[i] = y_train[closest_seq]
    return predictions
    
def NN_predict(double[:,:] x_train, int[:] y_train, double[:,:] x_test, int w):
    cdef int[:] predictions = CNN_predict(x_train, y_train, x_test, w)
    return np.asarray(predictions)
