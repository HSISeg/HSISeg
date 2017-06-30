import numpy as np
import scipy.io
from scipy import sparse
from sklearn import preprocessing
from numpy import linalg as LA
import image_helper as ih
import pickle,timeit,importlib
import numpy.matlib
from multiprocessing import Pool
from stable_simplex import threshold_u
import algo_default_params as default_params

def nonlocal_grad(W_sqrt, u):
	r = u.shape[0]
	diagu = sparse.lil_matrix((r,r),dtype=np.float64)
	diagu.setdiag(u)
	p1 = W_sqrt * diagu
	p2 = diagu * W_sqrt
	p = p1 - p2
	return p


def nonlocal_divergence(W_sqrt,v ):
	temp = W_sqrt.multiply(v)
	temp = temp - temp.transpose()
	div = temp.sum(axis=1)
	return np.array(div)[:,0]

def eu_distance(x, c):
	r,channel = x.shape
	cluster_no, channel = c.shape
	tempxc = np.matmul(x,c.transpose())
	tempx1 = np.repeat((((x**2).sum(axis=1)) * np.ones((1,1))).transpose(),cluster_no,axis=1)
	tempc2 = np.repeat(np.ones((1,1))*((c**2).sum(axis=1)),r,axis=0)
	dist = tempx1 + tempc2 - 2*tempxc
	return dist

def cond_projsplx_multi(Y,a_mat):
	n, m = Y.shape
	A = a_mat
	s = -np.sort(-(A*Y),axis=1)
	index = np.argsort(-(A*Y), axis=1)
	tmpsum = np.zeros(n) 
	tmpsumdom = np.zeros(n)
	bget = np.zeros(n, dtype=bool)
	A_sort = A[np.arange(np.shape(A)[0])[:,np.newaxis], index]
	cond_s = s/(A_sort**2)
	tmax = np.zeros(n)
	for ii in xrange(0,m-1):
		active = (bget==False)
		tmpsum[active] = tmpsum[active] + cond_s[active][:,ii]
		tmpsumdom[active] = tmpsumdom[active]+ 1.0/A_sort[active][:,ii]**2
		tmax[active] = (tmpsum[active] - 1)/tmpsumdom[active]
		deactivate = (tmax >= s[:,ii+1]) & active
		bget[deactivate] = True
	active = (bget==False)
	tmax[active] = (tmpsum[active] + cond_s[active][:,m-1] - 1)/(tmpsumdom[active]+1.0/(A_sort[active][:,m-1])**2)
	X = (Y - np.matlib.repmat(tmax.reshape(n,1),1,m)*1.0/A)
	X[X<0.0] = 0.0
	X = X/A
	return X

def project_p(p,r):
	tempp = sparse.csr_matrix(p, copy=False)
	tempp.data **= 2
	coe = np.array((tempp.sum(axis=1)))**0.5
	coe = np.amax(coe,axis=1)
	coe[coe<1.0] = 1
	diagcoe = sparse.lil_matrix((r,r),dtype=np.float64)
	diagcoe.setdiag(1/coe)
	p = diagcoe*p
	return p


def assing_classes(U,m,n):
	L = [[0 for i in xrange(n)] for j in xrange(m)]
	for j in xrange(0,n):
		for i in xrange(0,m):
			L[i][j] = U[j*m+i].argmax()
	return L

def calculate_centroid(uhard,cluster_no,endmem,image_2d):
	for l in xrange(0,cluster_no):
		index = np.nonzero(uhard[:,l])
		num_pixel = len(index[0])
		endmem[:,l] = (np.sum(image_2d[index],axis=0)/num_pixel)
	return endmem

def get_stop(uhard_old,uhard,r):
	iter_sparse = uhard_old - uhard
	stop = 1- (np.nonzero(iter_sparse)[0].shape[0])/r
	return stop

def calculate_cluster_wise_pdhg(params):
	u_bar_lth,sigma,tao,W_sqrt,l,r,p_lth,f_lth,u_lth = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
	p_lth = p_lth + nonlocal_grad(W_sqrt, sigma*u_bar_lth)
	p_lth = project_p(p_lth,r)
	u_lth = u_lth + nonlocal_divergence(W_sqrt, tao*p_lth) 
	u_lth = (1.0/(tao*f_lth+1)**0.5)*u_lth
	return [p_lth,u_lth]


def run_pdhg_quad(image,cluster_number,output_path,maxconn,params,pid_element):
	hsi_seg_algo = 'quad_pdhg'
	if params.get("weight_pickle_file_path"):
		W = ih.get_pickle_object_as_numpy(params.get("weight_pickle_file_path"))
	else:
		weight_params = {}
		if params.get('weight') and params['weight'].get('algo_params'):
			weight_params = params['weight'].get('algo_params')
		from pdhg_weight import pdhg_sparse_weight
		W = pdhg_sparse_weight(image,weight_params)
	pid_element.status_text = 'Weight calculated/Retreived'
	pid_element.percentage_done = '0'
	pid_element.save()

	if params.get("centroid_pickle_file_path"):
		endmem = ih.get_pickle_object_as_numpy(params.get("centroid_pickle_file_path"))
	else:
		if params.get("centroid_init") and params['centroid_init'].get('algo'):
			algo_name = params['centroid_init'].get('algo')
		else:
			algo_name = default_params.algo_details[hsi_seg_algo]['preprocessing']['centroid_init']['default_algo']
		centroid_init =  importlib.import_module('algo.centroid_init')
		algo = getattr(centroid_init,algo_name)
		endmem = algo(image,cluster_number)
	endmem = endmem.transpose()
	pid_element.status_text = 'Centroid calculated/Retreived'
	pid_element.percentage_done = '0'
	pid_element.save()

	algo_params = default_params.algo_details[hsi_seg_algo]['algo_params']
	if params.get('algo_params') and isinstance(params.get('algo_params'),dict):
		algo_params.update(params.get('algo_params'))
	pdhg_quad(image,W,algo_params['mu'],endmem,algo_params['lamda'],algo_params['tao'],algo_params['sigma'],algo_params['theta'],algo_params['iter_stop'],algo_params['innerloop'],algo_params['outerloop'],output_path,maxconn,pid_element)

def pdhg_quad(image,W,mu,endmem,lamda,tao,sigma,theta,iter_stop,innerloop,outerloop,output_path,maxconn,pid_element):
	image = image.astype(dtype=np.float64)
	m,n,channel = image.shape
	channel,cluster_no = endmem.shape
	uhard_old = np.zeros((m*n, cluster_no))
	stop = 0
	r = m*n
	u = np.ones((r,cluster_no))/cluster_no
	u_bar = np.array(u, copy=True)
	p = [sparse.csc_matrix((r,r),dtype=np.float64) for i in xrange(0,cluster_no)]
	count = 0
	diff = 1
	outer_index = 0
	error = np.zeros((innerloop*outerloop,1))
	image_2d = np.reshape(image, (r,channel),order='F')
	image_2d_normalized =  preprocessing.normalize(image_2d,norm='l2', axis=1)
	W_sqrt = W
	W.data **= 0.5
	colors = ih.generate_colors(cluster_no)
	while stop < iter_stop and count < innerloop*outerloop:
		pid_element.percentage_done = (count*100/(innerloop*outerloop))
		pid_element.status_text = 'Segmentation going on...'
		pid_element.save()
		outer_index += 1
		endmem_normalized = preprocessing.normalize(endmem,norm='l2', axis=0)
		temp = np.ones((r,cluster_no)) - np.matmul(image_2d_normalized,endmem_normalized) + mu * eu_distance(image_2d, endmem.transpose())**0.5
		f = 0.5*lamda*temp**2
		for jj in xrange(0,innerloop):
			count += 1
			uold = np.array(u, copy=True)
			data_inputs = [0 for i in xrange(0,cluster_no)]
			for l in xrange(0,cluster_no):
				data_inputs[l] = [u_bar[:,l],sigma,tao,W_sqrt,l,r,p[l],f[:,l],u[:,l]]
			pool = Pool(maxconn) # on 4 processors
			outputs = pool.map(calculate_cluster_wise_pdhg, data_inputs)
			pool.close()
			pool.join()
			for l in xrange(0,cluster_no):
				l_th_out = outputs[l]
				p[l] = l_th_out[0]
				u[:,l] = l_th_out[1]
			u = cond_projsplx_multi(u,(tao*f+1)**0.5)
			diff = LA.norm(u-uold,'fro')/LA.norm(u,'fro')
			u_bar = u+theta*(u-uold)
			error[count-1] = diff
		uhard = threshold_u(u.transpose(),maxconn)
		endmem = calculate_centroid(uhard,cluster_no,endmem,image_2d)
		L = assing_classes(uhard,m,n)
		ih.save_image(L,output_path + "_" + str(outer_index) + ".jpeg",colors)
		stop = get_stop(uhard_old,uhard,r)
		uhard_old = uhard
	return error
