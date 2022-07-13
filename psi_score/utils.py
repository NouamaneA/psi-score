import numpy as np
from scipy import sparse

def dict_to_sparse_matrix(d, shape, adj=False):
    row = []
    col = []
    data = []
    for i in d:
        row += len(d[i])*[i]
        col += list(d[i])
        if adj:
            data += len(d[i])*[1]
        else:
            data += list(d[i].values())
    return sparse.csr_matrix((data, (row, col)), shape=shape)

def l_plus_m(L, ls, ms):
    lpms = dict()
    for i in L.keys():
        lpms[i] = sum([ ls[l] + ms[l] for l in L[i] ])
    return lpms

def X(u, N):
    x = np.zeros(N)
    x[u] = 1
    return x