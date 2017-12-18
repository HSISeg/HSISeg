import numpy as np
import numpy.matlib
import random
from scipy import sparse
import scipy.io
import random,os,time,sys,math
import algo.image_helper as ih
from sklearn.mixture import GMM
from sklearn import mixture
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def mat_to_numpy(mat_file_path):
	mat = scipy.io.loadmat(mat_file_path)

def load_data():
	data = ih.get_pickle_object_as_numpy("data.pickle")
	data_gt = ih.get_pickle_object_as_numpy("data_gt.pickle")
	data_gt = data_gt['indian_pines_gt']
	data_cluster = ih.get_pickle_object_as_numpy("output_indian/output_20.pickle")
	data_cluster = np.asarray(data_cluster['L'])
	return data,data_gt,data_cluster

def reshape_data(data,data_gt,data_cluster):
	r,channel = data.shape[0]*data.shape[1],data.shape[2]
	X = np.reshape(data,(r,channel))
	Y = np.reshape(data_gt, (data_gt.shape[0]*data_gt.shape[1],))
	X_cluster = np.reshape(data_cluster,(data_cluster.shape[0]*data_cluster.shape[1],))
	X_tag_final = np.array(X_cluster, copy=True)
	return X,Y,X_cluster,X_tag_final


# random sample of 165 pixels for each class

def get_patch_end_points(data_gt,half_row,half_col):
	patch_end_points = [[] for i in xrange(0,data_gt.max()+1)]
	for i in xrange(0,data_gt.shape[0]):
		for j in xrange(0,data_gt.shape[1]):
			if len(patch_end_points[data_gt[i][j]]) == 0:
				start_row = i-half_row
				end_row = i+half_row
				if i-half_row < 0:
					start_row = 0
					end_row = end_row + (-(i-half_row)) if end_row + (-(i-half_row))<= data_gt.shape[0]-1 else end_row
				elif i+half_row > data_gt.shape[0]-1:
					end_row = data_gt.shape[0]-1
					start_row = start_row - (data_gt.shape[0]-1-i-half_row) if start_row - (data_gt.shape[0]-1-i-half_row)>=0 else start_row
				start_col = j-half_col
				end_col = j+half_col
				if j-half_col < 0:
					start_col = 0
					end_col = end_col + (-(j-half_col)) if end_col + (-(j-half_col))<= data_gt.shape[1]-1 else end_col
				elif j+half_col > data_gt.shape[1]-1:
					end_col = data_gt.shape[1]-1
					start_col = start_col - (data_gt.shape[1]-1-j-half_col) if start_col - (data_gt.shape[1]-1-j-half_col)>=0 else start_col
				my_patch = data_gt[start_row:end_row+1,:][:,start_col:end_col+1]
				class_pixels = np.where(my_patch == data_gt[i][j])
				purity = (class_pixels[1].shape[0]*100)/(my_patch.shape[0]*my_patch.shape[1]*1.0) 
				if(purity >= 85 and purity<95  ):
					patch_end_points[data_gt[i][j]] = [start_row,end_row,start_col,end_col]
	return patch_end_points

def save_train_image(data,indices_list,colors_class,output_path):
	n_classes = data.max()+1
	crop_img = np.array(data,copy = True)
	indices = None
	for i in xrange(0,len(indices_list)):
		if indices is None:
			indices = indices_list[i]
		else:
			indices = (np.append(indices[0],indices_list[i][0], axis = 0), np.append(indices[1],indices_list[i][1], axis=0))
	mask = np.ones(data.shape,dtype=bool) 
	mask[indices] = False
	crop_img[mask] = n_classes 
	colors_class.append([0,0,0])
	ih.save_image(crop_img,output_path,colors_class)
	return colors_class
	# X_train = None
	# Y_train = None
	# for i in xrange(0,Y.max()+1):

	# 	cluster_i = (Y==i)
	# 	cluster_i_data = X[cluster_i]
	# 	cluster_i_Y = Y[cluster_i]
	# 	n = ndatapt
	# 	if cluster_i_Y.shape[0] < ndatapt:
	# 		n = cluster_i_Y.shape[0]
	# 	index = np.random.choice(cluster_i_Y.shape[0], n, replace=False)
	# 	cluster_i_data = cluster_i_data[index]
	# 	cluster_i_Y = cluster_i_Y[index]
	# 	if X_train is None:
	# 		X_train = cluster_i_data
	# 		Y_train = cluster_i_Y
	# 	else:
	# 		X_train = np.append(X_train,cluster_i_data,axis = 0)
	# 		Y_train = np.append(Y_train,cluster_i_Y,axis = 0)

	# return X_train,Y_train

# def get_gmm_models(X_train,Y_train):
# 	gmm_models = []
# 	for i in xrange(0,Y_train.max()+1):
# 		cluster_i = (Y_train==i)
# 		cluster_i_data = X_train[cluster_i]
# 		gmm_model = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(cluster_i_data)
# 		gmm_models.append(gmm_model)
# 	return gmm_models

# def get_gmm_models_full_image(data,data_gt):
# 	gmm_models = []
# 	for i in xrange(0,data_gt.max()+1):
# 		cluster_i = (data_gt == i)
# 		X_train = data[cluster_i]	
# 		gmm_model = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(X_train_reshape)
# 		gmm_models.append(gmm_model)
# 	return gmm_models

def get_gmm_models(data,n_classes_gt,patch_indices):
	gmm_models = []
	for i in xrange(0,n_classes_gt):
		if len(patch_indices[i][0]) > 0:
			X_train = data[patch_indices[i]]
			gmm_model = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(X_train)
			gmm_models.append(gmm_model)
		else:
			gmm_models.append(None)
	return gmm_models

def shuffle_training(X_train,Y_train):
	#shuffle array
	assert len(X_train) == len(Y_train)
	shuffle_index = np.random.permutation(len(X_train))
	X_train = X_train[shuffle_index]
	Y_train = Y_train[shuffle_index]
	return X_train,Y_train


# def get_score_samples(gmm_model, X):
# 	prob = np.zeros(X.shape[0],1)
# 	for i in xrange(0,X.shape[0]):

def get_train_data(patch_indices,data,data_gt):
	all_rowi = patch_indices[0][0]
	all_coli = patch_indices[0][1]
	for i in xrange(1,len(patch_indices)):
		all_rowi = np.concatenate((all_rowi,patch_indices[i][0]))
		all_coli = np.concatenate((all_coli,patch_indices[i][1]))
	X_train = data[(all_rowi,all_coli)]
	Y_train = data_gt[(all_rowi,all_coli)]
	return X_train,Y_train

def get_roc_curves(X_train,Y_train,gmm_models,ndatapt):
	thresholds = []
	fprs = []
	tprs = []
	for i in xrange(0,len(gmm_models)):
		gmm_model = gmm_models[i]
		if gmm_model:
			#random select negative samples
			X_Y_zip = zip(X_train[Y_train!=i],Y_train[Y_train!=i])
			X_Y = random.sample(X_Y_zip,ndatapt)
			X_train_i = X_train[Y_train == i]
			Y_train_i = Y_train[Y_train == i]
			for j in xrange(0,ndatapt):
				X_train_i = np.append(X_train_i,np.reshape(X_Y[j][0],(1,X_Y[j][0].shape[0])),axis=0)
				Y_train_i = np.append(Y_train_i,X_Y[j][1])
			prob = gmm_model.score_samples(X_train_i)
			fpr, tpr, threshold = roc_curve(Y_train_i,prob,pos_label=i)
			fprs.append(fpr)
			tprs.append(tpr)
			thresholds.append(threshold)
		else:
			fprs.append(None)
			tprs.append(None)
			thresholds.append(None)
	return fprs,tprs,thresholds


def find_optimal_cutoff(fprs,tprs,thresholds):
	optimal_cutoff = []
	for j in xrange(0,len(thresholds)):
		fpr,tpr,threshold = fprs[j],tprs[j],thresholds[j]
		if (threshold is not None):
			i = np.arange(len(tpr))
			roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
			roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
			optimal_cutoff.append(list(roc_t['threshold'])[0])
		else:
			optimal_cutoff.append(None)
	return optimal_cutoff

def get_full_patch_indices(data_gt):
	patch_indices = []
	for i in xrange(0,data_gt.max()+1):
		patch_indices.append(np.where(data_gt==i))
	return patch_indices

def convert_patch_data_points(patch_end_points):
	patch_indices = []
	for i in xrange(0,len(patch_end_points)):
		if(len(patch_end_points[i]) == 0):
			patch_indices.append((np.array([],dtype='int64'),np.array([],dtype='int64')))
		else:
			row_size = patch_end_points[i][1] + 1 - patch_end_points[i][0]
			col_size = patch_end_points[i][3] + 1 - patch_end_points[i][2]
			total_points = row_size * col_size
			rows = np.zeros(total_points,dtype='int64')
			cols = np.zeros(total_points,dtype='int64')
			for k in xrange(patch_end_points[i][0],patch_end_points[i][1]+1):
				for l in xrange(patch_end_points[i][2],patch_end_points[i][3]+1):
					index = (k-patch_end_points[i][0])*col_size + l-patch_end_points[i][2]
					rows[index] = k
					cols[index] = l
			patch_indices.append((rows,cols))
	return patch_indices

def get_search_space(patch_indices,data_cluster,data_gt):
	search_space = []
	search_indices = []
	train_data = np.ones((data_cluster.shape[0],data_cluster.shape[1]),dtype=np.bool)
	search_image = np.array(data_gt,copy=True)
	for i in xrange(0,len(patch_indices)):
		if len(patch_indices[i][0]) > 0:
			patch_index = patch_indices[i]
			train_data[patch_index] = False
			search_space.append(np.unique(data_cluster[patch_index])) 
		else:
			search_space.append(None)
	search_image_bool = np.zeros((data_cluster.shape[0],data_cluster.shape[1]),dtype=np.bool)
	for i in xrange(0,len(search_space)):
		if search_space[i] is not None:
			mask = np.isin(data_cluster,search_space[i])
			search_mask = np.logical_and(mask,train_data)
			search_image_bool[search_mask] = True
			search_index = np.where(search_mask == True)
			search_indices.append(search_index)
		else:
			search_indices.append(None)
	left_index = np.where(search_image_bool==False)
	search_image[left_index] = data_gt.max()+1
	
	return search_indices,search_image

def clasify_cluster(data_cluster,gmm_models,data,optimal_cutoff,search_indices):
	X_tag_final = np.array(data_cluster, copy=True)
	X_tag_final.fill(-1)
	# X_tag_prob = np.array(data_cluster, copy=True)
	# X_tag_prob = np.zeros((data_cluster.shape[0],data_cluster.shape[1],len(search_space)))
	# X_tag_prob.fill(-np.inf)
	# X_tag_prob = [[(False,-np.inf) for i in xrange(0,len(search_space))]]
	X_tag_prob = np.zeros((data_cluster.shape[0],data_cluster.shape[1],len(search_indices)))
	X_tag_prob.fill(-np.inf)
	X_tagged = np.zeros((data_cluster.shape[0],data_cluster.shape[1],len(search_indices)),dtype=np.bool)

	# total_search_space = []
	# for i in xrange(0,len(search_space)):
	# 	if search_space[i] is not None:
	# 		total_search_space = np.union1d(total_search_space,search_space[i])
	# mask = np.isin(data_cluster,total_search_space)
	# search_space_pt_num = len(np.where(mask==True)[0])
	# print "search_space_points ", search_space_pt_num

	# for i in xrange(0,len(tag_cluster)):
	# 	X_tag_final[X_tag_final==tag_cluster[i]] = -2
	# X_tag_final[X_tag_final!=-2] = -1
	# print X_tag_final
	# for i in xrange(0,len(gmm_models)):
	# 	gmm_model = gmm_models[i]
	# 	if gmm_model is not None:
	# 		cluster_num = search_space[i]
	# 		mask = np.isin(data_cluster, cluster_num)
	# 		index = np.where(mask==True)
	# 		prob = gmm_model.score_samples(data[mask])
	# 		threshold = optimal_cutoff[i]
	# 		yes_index = np.where(prob > threshold) 
	# 		index = (index[0][yes_index],index[1][yes_index])
	# 		X_tag_prob[index] = prob[yes_index]
	# 		X_tag_final[index] = i 
			# for j in xrange():

	for i in xrange(0,len(gmm_models)):
		gmm_model = gmm_models[i]
		if gmm_model is not None:
			index = search_indices[i]
			prob = gmm_model.score_samples(data[index])
			threshold = optimal_cutoff[i]
			yes_index = np.where(prob> threshold)
			index = (index[0][yes_index],index[1][yes_index])	
			index3 = np.zeros(index[0].shape[0],dtype='int64')
			index3.fill(i)
			index = (index[0],index[1],index3)
			X_tag_prob[index] = prob[yes_index]
			X_tagged[index] = True	

	for i in xrange(0,X_tag_final.shape[0]):
		for j in xrange(0,X_tag_final.shape[1]):
			tag = -1
			prob = -np.inf
			for k in xrange(0, len(gmm_models)):
				if X_tagged[i][j][k] == True:
					if X_tag_prob[i][j][k]>prob:
						tag = k
						prob = X_tag_prob[i][j][k]

			X_tag_final[i][j] = tag

	return X_tag_final
# image_2d_normalized =  preprocessing.normalize(image_2d,norm='l2', axis=1)

def get_train_indices(data_gt,patch_half_row,patch_half_col):
	patch_end_points = get_patch_end_points(data_gt,patch_half_row,patch_half_col)
	patch_indices = convert_patch_data_points(patch_end_points)
	# patch_indices = get_full_patch_indices(data_gt)
	return patch_indices

def save_classified_image(X_tag_final,output_indian):
	unique_class = np.unique(X_tag_final)
	n_classes = unique_class.shape[0]
	tag_img = np.array(X_tag_final, copy=True)
	colors = ih.generate_colors(n_classes)
	if -1 in unique_class:
		colors[0] = [0,0,0]
	for i in xrange(0, unique_class.shape[0]):
		tag_img[np.where(X_tag_final==unique_class[i])] = i
	ih.save_image(tag_img,output_indian,colors)

def get_precision_recall(X_tag_final,data_gt):
	n_classes = data_gt.max() + 1
	data_gt_reshape = np.reshape(data_gt,(data_gt.shape[0]*data_gt.shape[1]))
	X_tag_final_reshape = np.reshape(X_tag_final,(X_tag_final.shape[0]*X_tag_final.shape[1]))
	data_gt_reshape[-1] = n_classes
	X_tag_final_reshape[np.where(X_tag_final_reshape == -1)] = n_classes
	precisions = precision_score(data_gt_reshape,X_tag_final_reshape,average=None)
	recalls = recall_score(data_gt_reshape, X_tag_final_reshape, average=None)
	return precisions, recalls

# ndatapt = 150
patch_half_row = 6
patch_half_col = 6
ndatapt = int((patch_half_row*2+1)*(patch_half_col*2+1)*0.85)
data,data_gt,data_cluster = load_data()
n_classes_gt = data_gt.max()+1
colors_class = ih.generate_colors(n_classes_gt)
ih.save_image(data_gt,"output_indian/Anirban/indian_pines_gt.jpeg",colors_class)
#list of indices for training data for each class
patch_indices = get_train_indices(data_gt,patch_half_row,patch_half_col)
colors_class = save_train_image(data_gt,patch_indices,colors_class,"output_indian/Anirban/train_data.jpeg")
#list of gmm models for each class
gmm_models = get_gmm_models(data,n_classes_gt,patch_indices)
#mixture of training data
X_train, Y_train = get_train_data(patch_indices,data,data_gt)
#list of false positive , true positive and thresholds for roc curve of each class (based on gmm classifier)
fprs,tprs,thresholds = get_roc_curves(X_train,Y_train,gmm_models,ndatapt)
# optimal threshold for each class
optimal_cutoff = find_optimal_cutoff(fprs,tprs,thresholds)
#list of cluster number that needs to be searched for each class
search_indices,search_image = get_search_space(patch_indices,data_cluster,data_gt)
ih.save_image(search_image,"output_indian/Anirban/search_space.jpeg",colors_class)
# print "search_space = ", search_space
#classify cluster
X_tag_final = clasify_cluster(data_cluster,gmm_models,data,optimal_cutoff,search_indices)
save_train_image(data_gt,[np.where(X_tag_final!=-1)],colors_class[:-1],"output_indian/Anirban/classified_gt.jpeg")
save_classified_image(X_tag_final,"output_indian/Anirban/classified_cluster.jpeg")
precisions, recalls = get_precision_recall(X_tag_final,data_gt)

print "percentage of points classified ",len(np.where(X_tag_final!=-1)[0])*100/(data_gt.shape[0]*data_gt.shape[1])
index = np.where(X_tag_final!=-1)

correctly_classified = len(np.where(data_gt[index] == X_tag_final[index])[0])
print "percentage of points correctly classified ",correctly_classified*100/len(np.where(X_tag_final!=-1)[0])

print "precisions", precisions
print "recalls", recalls



# colors_class = ih.generate_colors(n_classes_gt)
# colors_cluster = ih.generate_colors(data_cluster.max()+1)
# ih.save_image(data_gt,"output_indian/indian_pines_gt.jpeg",colors_class)
# ih.save_image(data_cluster,"output_indian/indian_pines_cluster.jpeg",colors_cluster)
# gmm_models = get_gmm_models(data,data_gt,patch_end_points)

# X,Y,X_cluster,X_tag_final = reshape_data(data,data_gt,data_cluster)
# gmm_models = get_gmm_models(data,Y_train)
# X_train,Y_train = shuffle_training(X_train,Y_train)
# fprs,tprs,thresholds = get_roc_curves(X_train,Y_train,gmm_models,ndatapt)
# optimal_cutoff = find_optimal_cutoff(fprs,tprs,thresholds)




# n_classes = Y_train.max()+1
# colors_gt = ih.generate_colors(Y.max()+1)
# ih.save_image(data_gt,"output_indian/ouput_gt.jpeg",colors_gt)