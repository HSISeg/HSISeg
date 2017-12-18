import numpy as np
import numpy.matlib
from scipy import sparse
import random,os,time,sys,math
import algo.image_helper as ih
from multiprocessing import Pool
import fuzzy_beta
import importlib
import algo.algo_default_params as default_params
from sklearn.mixture import GMM
from sklearn import mixture
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from scipy.stats import multivariate_normal

data = ih.get_pickle_object_as_numpy("data.pickle")
data_gt = ih.get_pickle_object_as_numpy("data_gt.pickle")
data_gt = data_gt['indian_pines_gt']
data_my = ih.get_pickle_object_as_numpy("output_indian/output_20.pickle")
data_my = np.asarray(data_my['L'])
r,channel = data.shape[0]*data.shape[1],data.shape[2]
X = np.reshape(data,(r,channel))
Y = np.reshape(data_gt, (data_gt.shape[0]*data_gt.shape[1],))
X_train = None
Y_train = None
# model = gnb.fit(X, Y)
X_my = np.reshape(data_my,(data_my.shape[0]*data_my.shape[1],))
X_my_final = np.array(X_my, copy=True)
muln_dist = {}
prior_prob = [1.0/(Y.max()+1) for i in xrange(0,Y.max()+1)]
gmm_models = []

# random sample of 150 pixels for each class
for i in xrange(0,Y.max()+1):
	cluster_i = (Y==i)
	cluster_i_data = X[cluster_i]
	cluster_i_Y = Y[cluster_i]
	n = 150
	if cluster_i_Y.shape[0] < 150:
		n = cluster_i_Y.shape[0]
	index = np.random.choice(cluster_i_Y.shape[0], n, replace=False)
	cluster_i_data = cluster_i_data[index]
	cluster_i_Y = cluster_i_Y[index]
	mean = np.mean(cluster_i_data, axis=0)
	cov = np.cov(cluster_i_data, rowvar=0)
	muln_dist[i] = [mean,cov]
	gmm_model = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(cluster_i_data)
	gmm_models.append(gmm_model)
	# y_more = np.random.multivariate_normal(mean, cov)
	# muln_dist.append(y_more)
	if X_train is None:
		X_train = cluster_i_data
		Y_train = cluster_i_Y
	else:
		X_train = np.append(X_train,cluster_i_data,axis = 0)
		Y_train = np.append(Y_train,cluster_i_Y,axis = 0)
		

#shuffle array
assert len(X_train) == len(Y_train)
shuffle_index = np.random.permutation(len(X_train))
X_train = X_train[shuffle_index]
Y_train = Y_train[shuffle_index]




n_classes = Y_train.max()+1
colors_gt = ih.generate_colors(Y.max()+1)
ih.save_image(data_gt,"output_indian/ouput_gt.jpeg",colors_gt)


# GMM Multivariate Classifier
print 'Running GMM Multivariate Classifier'
for i in xrange(0,X_my.max()+1):
	cluster_i = (X_my == i)
	data_to_predict = X[cluster_i]
	

	for j in xrange(0,X_my.max()+1):
		post_prob_comp = gmm_models[i].predict_proba(data_to_predict)
		
		gmm_models[j].predict(data_to_predict)

		prob_val = prior_prob[j]*multivariate_normal.pdf(data_to_predict,mean=muln_dist[i][0], cov=muln_dist[i][1],allow_singular=True)



# GaussianNB classifier
print 'Running GaussianNB'
gnb = GaussianNB()
model = gnb.fit(X_train, Y_train)
for i in xrange(0,X_my.max()+1):
	cluster_i = (X_my == i)
	data_to_predict = X[cluster_i]
	pred = classifier.predict(data_to_predict)
	this_class = np.argmax(np.bincount(pred))
	X_my_final[cluster_i] = this_class
X_my_final_data = np.reshape(X_my_final,(data_my.shape[0],data_my.shape[1]))
colors = ih.generate_colors(X_my_final_data.max()+1)
ih.save_image(X_my_final_data,"output_indian/GaussianNB_ouput.jpeg",colors)
print (np.count_nonzero(X_my_final_data == data_gt)* 100)/(data_gt.shape[0]*data_gt.shape[1]),(np.count_nonzero(X_my_final_data != data_gt)* 100)/(data_gt.shape[0]*data_gt.shape[1]),'GaussianNB'


# GMM Classifier
print 'Running GMM' 
classifiers = dict((covar_type, GMM(n_components=n_classes,
                    covariance_type=covar_type, init_params='wc', n_iter=20))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])
for index, (name, classifier) in enumerate(classifiers.items()):
	classifier.means_ = np.array([X_train[Y_train == i].mean(axis=0)
	                                  for i in xrange(n_classes)])
	classifier.fit(X_train)
	for i in xrange(0,X_my.max()+1):
		cluster_i = (X_my == i)
		data_to_predict = X[cluster_i]
		pred = classifier.predict(data_to_predict)
		this_class = np.argmax(np.bincount(pred))
		X_my_final[cluster_i] = this_class
	X_my_final_data = np.reshape(X_my_final,(data_my.shape[0],data_my.shape[1]))
	colors = ih.generate_colors(X_my.max()+1)
	ih.save_image(X_my_final_data,"output_indian/GMM_ouput_"+name+".jpeg",colors)
	print (np.count_nonzero(X_my_final_data == data_gt)* 100)/(data_gt.shape[0]*data_gt.shape[1]),(np.count_nonzero(X_my_final_data != data_gt)* 100)/(data_gt.shape[0]*data_gt.shape[1]),name




def get_initial_u(data, cluster_number, V):
	row_size = data.shape[0]
	col_size = data.shape[1]
	U = np.random.rand(row_size,col_size,cluster_number)
	for i in xrange(0,row_size):
		for j in xrange(0,col_size):
			z = (((np.matlib.repmat(data[i][j],cluster_number,1) - V)**2).sum(axis=1)) ** 0.5
			z = z/np.sum(z)
			z_exp = [math.exp(-k) for k in z]  
			sum_z_exp = sum(z_exp)  
			softmax = [round(k / sum_z_exp, 3) for k in z_exp]
			U[i][j] = np.array(softmax)
	return U


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
		pid_element.percentage_done = ((n-1)*100/max_iter)
		pid_element.status_text = 'Segmentation going on...'
		pid_element.save()
		V = get_cluster_prototypes(U,data,m,cluster_number)
		ih.save_output(L_new,V,output_path + "_" + str(n-1) + ".pickle")
		D = get_dissimilarity_matrix(U,V,data,n,error_list,beta,alpha_w,alpha_e_avg_t,alpha_n0,maxconn)
		U = update_U(U,D,m)
		L = L_new
		L_new = assing_classes(U)
		ih.save_image(L_new,output_path + "_" + str(n) + ".jpeg",colors)
		mean_error = get_segmentation_error(L,L_new,data)
		error_list.append(mean_error)

		if mean_error < terminating_mean_error:
			break
		n += 1
	ih.save_output(L_new,V,output_path + "_" + str(n) + ".pickle")
	return
