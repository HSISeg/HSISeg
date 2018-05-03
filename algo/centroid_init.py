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


def random_centroids(data,cluster_number):
	row_size = data.shape[0]
	col_size = data.shape[1]
	channel_count = data.shape[2]
	m = 1.3
	U = np.random.rand(row_size,col_size,cluster_number)
	for i in range(0,row_size):
		for j in range(0,col_size):
			total_u = sum(U[i][j])
			U[i][j] / total_u
	V = np.zeros((cluster_number,channel_count))
	for r in range(0,cluster_number):
		normalizer = 0.0
		cluster_center = np.zeros(channel_count)
		for i in range(0,row_size):
			for j in range(0,col_size):
				cluster_center += (U[i][j][r]**m) * data[i][j]
				normalizer += U[i][j][r]**m
		V[r] = cluster_center / normalizer
		
	return V