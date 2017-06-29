import itertools
import numpy as np
from scipy import sparse
from scipy import signal
import algo_default_params as default_params

def beta_spatial(f,params):
    row_size = f.shape[0]
    col_size = f.shape[1]
    r = row_size*col_size
    algo_params = default_params.fuzzy_c_means_beta_algo['beta_spatial']['algo_params']
    if isinstance(params,dict):
        algo_params.update(params)
    max_points = algo_params['max_points']
    theta = algo_params['theta']

    e = 2.71828183
    all_points = np.arange(0,r)
    all_points_row = all_points/col_size
    all_points_col = all_points%col_size
    cols = [[] for i in xrange(0,r)]
    rows = [[] for i in xrange(0,r)]
    values = [[] for i in xrange(0,r)]
    for i in xrange(0,r):
        i_row,i_col = i/col_size,i%col_size
        values_i = (abs(all_points_row - i_row) +  abs(all_points_col-i_col))
        values_i = 1.0/(1.0 + e**(theta*values_i))
        values_i[values_i<1e-7] = 0
        col_indexes = np.nonzero(values_i>0)[0]
        if max_points !=-1 and col_indexes.shape > max_points:
            col_indexes = np.argsort(values_i)[::-1][:max_points]
        cols[i] = (col_indexes).tolist()
        row_indexes = np.zeros(col_indexes.shape[0],dtype=np.int32)+i
        rows[i] = row_indexes.tolist()
        values[i] = (values_i[col_indexes]).tolist()
        del values_i
        del col_indexes
        del row_indexes
    cols = list(itertools.chain(*cols))
    rows = list(itertools.chain(*rows))
    values = list(itertools.chain(*values))
    beta = sparse.csr_matrix((values,(rows,cols)),shape = (r,r))
    return beta

def gaussian_filter(shape =(5,5), sigma=1):
    x, y = [edge /2 for edge in shape]
    grid = np.array([[((i**2+j**2)/(2.0*sigma**2)) for i in xrange(-x, x+1)] for j in xrange(-y, y+1)])
    g_filter = np.exp(-grid)/(2*np.pi*sigma**2)
    g_filter /= np.sum(g_filter)
    return g_filter

def beta_spectral_spatial(f,params):
    algo_params = default_params.fuzzy_c_means_beta_algo['beta_spectral_spatial']['algo_params']
    if isinstance(params,dict):
        algo_params.update(params)
    ms = algo_params['max_points']
    ws = algo_params['half_search_window']
    ps = algo_params['half_patch_size']
    sigma = algo_params['gaussian_sigma']
    mu = algo_params['euclidean_distance_weight']
    theta = algo_params['theta']

    f = f.astype(dtype=np.float64)
    m, n, channel = f.shape
    r = m*n
    e = 2.71828183
    G = gaussian_filter(shape=(2*ps+1,2*ps+1),sigma=sigma)
    dist = np.zeros(((2*ws+1)*(2*ws+1), r))
    pad_width = ((ws,ws),(ws,ws),(0,0))
    padu = np.lib.pad(f,pad_width=pad_width,mode='symmetric',reflect_type='even')
    for i in xrange(-ws,ws+1):
        for j in xrange(-ws,ws+1):
            pad_width = ((ws-i,ws+i),(ws-j,ws+j),(0,0))
            shiftpadu = np.lib.pad(f,pad_width=pad_width,mode='symmetric',reflect_type='even')  
            temp1 = (padu*shiftpadu).sum(axis=2)
            temp2 = (((padu**2).sum(axis=2))**0.5) *(((shiftpadu**2).sum(axis=2))**0.5)
            tempu = 1 - temp1/temp2 + mu * (((padu-shiftpadu)**2).sum(axis=2)**0.5)
            padtempu = tempu[ws-ps:m+ws+ps, ws-ps:n+ws+ps]
            uu = signal.convolve2d(padtempu**2,G,'same')
            uu = uu[ps:m+ps, ps:n+ps]
            k = (j+ws)*(2*ws+1)+i+ws
            dist[k,:] = np.reshape(uu, (1,r))[0]
    W = sparse.csr_matrix((r,r),dtype=np.float64)
    idx = np.arange(0,r)
    dist[~np.isfinite(dist)] = np.inf
    dist[dist<1e-13] = 1e+5
    dist = 1.0/(1.0 + e**(theta*dist))
    for i in xrange(0,ms):
        minindex = np.argmax(dist,axis=0)
        y = np.max(dist,axis=0)
        indexes_to_set = (minindex,idx)
        ind1 = np.arange(0,r).reshape(r,1)
        minindex = minindex.reshape(r,1)
        ind2 = np.floor((minindex)/(2*ws+1))*(m-2*ws-1) + minindex +ind1 -ws - ws*m
        tmpindex = np.intersect1d(np.nonzero(ind2>=0)[0], np.nonzero(ind2<r)[0])
        cols = ind2[tmpindex].astype(np.int32)
        rows = ind1[tmpindex]
        values = y[tmpindex]
        W = W + sparse.csr_matrix((values,(rows.transpose()[0],cols.transpose()[0])),shape = (r,r))
        dist[indexes_to_set] = -np.inf
        
    return W
    
