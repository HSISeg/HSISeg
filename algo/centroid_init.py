import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq, kmeans, whiten

def scipy_kmeans_centroids(data,cluster_number):
	data = data.astype(dtype=np.float64)
	r,channel = data.shape[0]*data.shape[1],data.shape[2]
	X = np.reshape(data,(r,channel))
	kmeans_res = kmeans(X,cluster_number)
	return kmeans_res[0].astype(dtype=np.float64) 

def scklearn_kmeans_centroids(data,cluster_number):
	data = data.astype(dtype=np.float64)
	r,channel = data.shape[0]*data.shape[1],data.shape[2]
	X = np.reshape(data,(r,channel))
	kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(X)
	centroids = kmeans.cluster_centers_
	return centroids.astype(dtype=np.float64)