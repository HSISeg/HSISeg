import numpy as np
import numpy.matlib
from scipy import sparse
import random,os,time,sys,math
from . import image_helper as ih
from multiprocessing import Pool
from . import fuzzy_beta
import importlib
from . import algo_default_params as default_params

def get_initial_u(data, cluster_number, V):
	row_size = data.shape[0]
	col_size = data.shape[1]
	U = np.random.rand(row_size,col_size,cluster_number)
	for i in range(0,row_size):
		for j in range(0,col_size):
			z = (((np.matlib.repmat(data[i][j],cluster_number,1) - V)**2).sum(axis=1)) ** 0.5
			z = z / np.sum(z)
			z_exp = [math.exp(-k) for k in z]  
			sum_z_exp = sum(z_exp)  
			softmax = [round(k / sum_z_exp, 3) for k in z_exp]
			U[i][j] = np.array(softmax)
	return U


def get_cluster_prototypes(U,data,m,cluster_number):
	row_size = data.shape[0]
	col_size = data.shape[1]
	channel_count = data.shape[2]
	V = np.zeros((cluster_number,channel_count))
	U_new = U.reshape(row_size*col_size,cluster_number)
	data_new = data.reshape(row_size*col_size,channel_count)
	normalizer = np.sum(U**m,axis=(0,1))
	for r in range(0,cluster_number):
		V[r] = ((np.matlib.repmat(U_new[:,r]**m,channel_count,1).transpose() * data_new).sum(axis=0)) / normalizer[r]
	return V

def get_alpha(n,error_list,alpha_w,alpha_e_avg_t,alpha_n0):
	e = 2.71828183
	if len(error_list) > 6:
		avg_error = sum(error_list[-6:]) / 6
		if avg_error < alpha_e_avg_t:
			n0 = n
		else:
			n0 = alpha_n0
	else:
		n0 = alpha_n0
	return 0.2 / (0.1 + e**((n0-n) / alpha_w))

def get_segmentation_error(L,L_new,data):
	row_size = data.shape[0]
	col_size = data.shape[1]

	total_error = 0.0
	count = 0
	for i in range(0,row_size):
		for j in range(0,col_size):
			if L_new[i][j] != L[i][j]:
				total_error += 1.0
			count += 1
	return total_error / float(count)


def get_feature_dissimilarity(X,x,y,V,r,channel_count):
	s = np.sum((X[x][y] - V[r])**2,axis=0)
	return s**0.5

def compute_cluster_distances_pool(params):
	U_new,V,X_x_y,x,y,alpha,beta_x_y = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
	cluster_number = U_new.shape[1]
	normalizer = 0.0
	D = np.sum(U_new.transpose() * np.matlib.repmat(beta_x_y.toarray()[0],cluster_number,1),axis=1)
	D = (1.0 - (D / np.sum(D))) * alpha + np.sum((np.matlib.repmat(X_x_y,cluster_number,1) - V)**2,axis=1)**0.5
	return D

def compute_cluster_distances(U,V,X,D,x,y,alpha,beta):
	row_size = X.shape[0]
	col_size = X.shape[1]
	channel_count = X.shape[2]
	cluster_number = U.shape[2]
	normalizer = 0.0
	U_new = U.reshape(row_size*col_size,cluster_number)
	D[x][y] = np.sum(U_new.transpose() * np.matlib.repmat(beta[x*row_size+y,:].toarray()[0],cluster_number,1),axis=1)
	D[x][y] = (1.0 - (D[x][y] / np.sum(D[x][y]))) * alpha + np.sum((np.matlib.repmat(X[x][y],cluster_number,1) - V)**2,axis=1)**0.5
	return 


def get_dissimilarity_matrix(U,V,X,n,error_list,beta,alpha_w,alpha_e_avg_t,alpha_n0,maxconn):
	row_size = X.shape[0]
	col_size = X.shape[1]
	channel_count = X.shape[2]
	alpha = get_alpha(n,error_list,alpha_w,alpha_e_avg_t,alpha_n0)
	cluster_number = V.shape[0]
	D = np.zeros((row_size,col_size,cluster_number)) 
	index_arr = np.array([[k,l] for k in range(row_size) for l in range(col_size)],dtype='int32')
	U_new = U.reshape(row_size*col_size,cluster_number, order='F')
	data_inputs = [0 for i in range(0,row_size*col_size)]
	for i in range(0, row_size*col_size):
		x = index_arr[i][0]
		y = index_arr[i][1]
		data_inputs[i] = [U_new,V,X[x][y],x,y,alpha,beta[x*row_size+y,:]]
	pool = Pool(maxconn) 
	outputs = pool.map(compute_cluster_distances_pool, data_inputs)
	pool.close()
	pool.join()
	for i in range(0,row_size*col_size):
		x = index_arr[i][0]
		y = index_arr[i][1]
		D[x][y] = outputs[i]
	return D

def update_U(U,D,m):
	row_size = U.shape[0]
	col_size = U.shape[1]
	cluster_number = U.shape[2]
	min_distance = 0.00000001

	for i in range(0,row_size):
		for j in range(0,col_size):
			good_classes = [c for c in range(0,cluster_number) if D[i][j][c] <= min_distance]
			if len(good_classes) > 0:
				for r in range(0,cluster_number):
					U[i][j][r] = 0.0
				for r in good_classes:
					U[i][j][r] = 1.0
			else:
				for r in range(0,cluster_number):
					U[i][j][r] = 1.0 / sum(    [( D[i][j][r] / x )**(2 / (m-1)) for x in D[i][j]]   )
	for i in range(0,row_size):
		for j in range(0,col_size):
			normalizer = sum(U[i][j])
			for r in range(0,cluster_number):
				U[i][j][r] = U[i][j][r] / normalizer
	return U

def assing_classes(U):
	row_size = U.shape[0]
	col_size = U.shape[1]
	cluster_number = U.shape[2]
	L = [[0 for j in range(0,col_size)] for i in range(0,row_size)]
	for i in range(0,row_size):
		for j in range(0,col_size):
			L[i][j] = U[i][j].argmax()
	return L


def run_fuzzy_c_means(image,cluster_number,output_path,maxconn,params,pid_element):
	hsi_seg_algo = 'fuzzy_c_means'
	
	if params.get("beta_pickle_file_path"):
		beta = ih.get_pickle_object_as_numpy(params.get("beta_pickle_file_path"))
	else:
		beta_algo_name = default_params.algo_details['fuzzy_c_means']['preprocessing']['beta']['default_algo']
		if params.get('beta') and params['beta'].get('algo'):
			beta_algo_name = params['beta'].get('algo')
		beta_algo_params = {}
		if params.get('beta') and params['beta'].get('algo_params'):
			beta_algo_params = params['beta'].get('algo_params')
		beta_algo = getattr(fuzzy_beta,beta_algo_name)
		beta = beta_algo(image,beta_algo_params)
	pid_element.status_text = 'Beta calculated/Retreived'
	pid_element.percentage_done = '0'
	pid_element.save()
	
		
	if params.get("centroid_pickle_file_path"):
		initial_centroids = ih.get_pickle_object_as_numpy(params.get("centroid_pickle_file_path"))
	else:
		if params.get("centroid_init") and params['centroid_init'].get('algo'):
			algo_name = params['centroid_init'].get('algo')
		else:
			algo_name = default_params.algo_details[hsi_seg_algo]['preprocessing']['centroid_init']['default_algo']
		centroid_init =  importlib.import_module('algo.centroid_init')
		algo = getattr(centroid_init,algo_name)
		initial_centroids = algo(image,cluster_number)
	pid_element.status_text = 'Centroid calculated/Retreived'
	pid_element.percentage_done = '0'
	pid_element.save()

	algo_params = default_params.algo_details[hsi_seg_algo]['algo_params']
	if params.get('algo_params') and isinstance(params.get('algo_params'),dict):
		algo_params.update(params.get('algo_params'))

	fuzzy_c_means(image,beta,initial_centroids,algo_params['fuzzy_index'],algo_params['terminating_mean_error'],algo_params['alpha_w'],algo_params['alpha_e_avg_t'],algo_params['alpha_n0'],algo_params['max_iter'],output_path,maxconn,pid_element)


def fuzzy_c_means(data,beta,initial_centroids,m,terminating_mean_error,alpha_w,alpha_e_avg_t,alpha_n0,max_iter,output_path,maxconn,pid_element):
	row_size = data.shape[0]
	col_size = data.shape[1]
	cluster_number = initial_centroids.shape[0]
	##### initializing ###########
	U = get_initial_u(data,cluster_number,initial_centroids)
	error_list = []
	L_new = assing_classes(U)
	colors = ih.generate_colors(cluster_number)
	ih.save_image(L_new,output_path + "_" + str(0) + ".jpeg",colors)
	##### starting iterations ####
	n = 1

	while n <= max_iter:
		pid_element.percentage_done = ((n-1)*100 / max_iter)
		pid_element.status_text = 'Segmentation going on...'
		pid_element.save()
		V = get_cluster_prototypes(U,data,m,cluster_number)
		# ih.save_output(L_new,V,output_path + "_" + str(n-1) + ".pickle")
		D = get_dissimilarity_matrix(U,V,data,n,error_list,beta,alpha_w,alpha_e_avg_t,alpha_n0,maxconn)
		U = update_U(U,D,m)
		L = L_new
		L_new = assing_classes(U)
		ih.save_image(L_new,output_path + "_" + str(n) + ".jpeg",colors)
		ih.save_output_dict({'L': L, 'cluster_centres': V, 'U': U}, output_path + "_data_" + str(n-1) + ".pickle")
		mean_error = get_segmentation_error(L,L_new,data)
		error_list.append(mean_error)

		if mean_error < terminating_mean_error:
			break
		n += 1
	ih.save_output(L_new,V,output_path + "_" + str(n) + ".pickle")
	return
