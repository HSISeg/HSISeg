import numpy as np
import scipy.io
from scipy import sparse
from scipy import signal
from . import algo_default_params as default_params

def gaussian_filter(shape =(5,5), sigma=1):
	x, y = [edge // 2 for edge in shape]
	grid = np.array([[((i**2+j**2) // (2.0*sigma**2)) for i in range(-x, x+1)] for j in range(-y, y+1)])
	g_filter = np.exp(-grid)/(2*np.pi*sigma**2)
	g_filter /= np.sum(g_filter)
	return g_filter

def pdhg_sparse_weight(f,params):
	weight_params = default_params.weight_algo['pdhg_sparse_weight']['algo_params']
	if isinstance(params,dict):
		weight_params.update(params)
	ms = weight_params['max_points']
	ws = weight_params['half_search_window']
	ps = weight_params['half_patch_size']
	sigma = weight_params['gaussian_sigma']
	mu = weight_params['euclidean_distance_weight']


	f = f.astype(dtype=np.float64)
	m, n, channel = f.shape
	r = m*n
	G = gaussian_filter(shape=(2*ps+1,2*ps+1),sigma=sigma)
	dist = np.zeros(((2*ws+1)*(2*ws+1), r))
	pad_width = ((ws,ws),(ws,ws),(0,0))
	padu = np.lib.pad(f,pad_width=pad_width,mode='symmetric',reflect_type='even')
	for i in range(-ws,ws+1):
		for j in range(-ws,ws+1):
			pad_width = ((ws-i,ws+i),(ws-j,ws+j),(0,0))
			shiftpadu = np.lib.pad(f,pad_width=pad_width,mode='symmetric',reflect_type='even')  
			temp1 = (padu*shiftpadu).sum(axis=2)
			temp2 = (((padu**2).sum(axis=2))**0.5) *(((shiftpadu**2).sum(axis=2))**0.5)
			tempu = 1 - temp1/temp2 + mu * (((padu-shiftpadu)**2).sum(axis=2)**0.5)
			padtempu = tempu[ws-ps:m+ws+ps, ws-ps:n+ws+ps]
			uu = signal.convolve2d(padtempu**2,G,'same')
			uu = uu[ps:m+ps, ps:n+ps]
			k = (j+ws)*(2*ws+1)+i+ws
			dist[k,:] = np.reshape(uu, (1,r), order='F')[0]
	W = sparse.csr_matrix((r,r),dtype=np.float64)
	idx = np.arange(0,r)
	dist[~np.isfinite(dist)] = np.inf
	dist[dist<1e-13] = 1e+5
	for i in range(0,ms):
		minindex = np.argmin(dist,axis=0)
		indexes_to_set = (minindex,idx)
		ind1 = np.arange(0,r).reshape(r,1)
		minindex = minindex.reshape(r,1)
		ind2 = np.floor((minindex)/(2*ws+1))*(m-2*ws-1) + minindex +ind1 -ws - ws*m
		tmpindex = np.intersect1d(np.nonzero(ind2>=0)[0], np.nonzero(ind2<r)[0])
		cols = ind2[tmpindex].astype(np.int32)
		rows = ind1[tmpindex]
		values = np.ones(len(tmpindex))
		W = W + sparse.csr_matrix((values,(rows.transpose()[0],cols.transpose()[0])),shape = (r,r))
		dist[indexes_to_set] = np.inf
	W = W.tocsc()
	return W