from PIL import Image
import numpy as np
import pickle
import scipy
import colorsys

def mat_to_numpy(mat_file_path):
	mat = scipy.io.loadmat(mat_file_path)


def create_test_image(image_path,row,col):
	channel = 3
	image_matrix = [[[0 for x in range(0,channel)] for y in range(0,col)] for z in range(0,row)]
	for x in range(0,row):
		for y in range(0,col):
			if x <= row // 2 and y <= col // 2:
				image_matrix[x][y] = [255,0,0]
			elif x <= row // 2 and y > col // 2:
				image_matrix[x][y] = [0,0,255]
			elif x > row // 2 and y <= col // 2:
				image_matrix[x][y] = [0,255,0]
			else:
				image_matrix[x][y] = [0,0,0]
	image_matrix_np = np.array(image_matrix, dtype='uint8')
	im = Image.fromarray(image_matrix_np)
	im.save(image_path)
	return

def get_data_from_image(image_path):
# 	from osgeo import gdal
	from skimage import io
	if image_path.split(".")[1] == "tif":
		M = io.imread(image_path)
# 		dataset = gdal.Open(image_path,gdal.GA_ReadOnly)
# 		col = dataset.RasterXSize
# 		row = dataset.RasterYSize
# 		a = [[[]for y in xrange(col)] for z in xrange(row)]
# 		for i in xrange(1,dataset.RasterCount + 1):
# 			band = dataset.GetRasterBand(i).ReadAsArray()
# 			for m in xrange(0,row):
# 				for n in xrange(0,col):
# 					a[m][n].append(band[m][n])
# 		M = np.array(a,dtype='uint16')
	else:
		M = np.asarray(Image.open(image_path))
	return M


def save_image_as_pickle(image_path):
	M = np.asarray(Image.open(image_path), dtype=np.float64)
	image_name = image_path.split(".")[0]
	pickle_file_name = image_name + ".pickle"
	with open(pickle_file_name, "wb") as fp:
		pickle.dump(M, fp, protocol=pickle.HIGHEST_PROTOCOL)
	return

def generate_colors(cluster_number):
	HSV_tuples = [(x*1.0 / cluster_number, 0.5, 0.5) for x in range(cluster_number)]
	RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
	for i in range(0,len(RGB_tuples)):
		RGB_tuples[i] = [int(RGB_tuples[i][0]*256),int(RGB_tuples[i][1]*256),int(RGB_tuples[i][2]*256)]
	return RGB_tuples

def get_pickle_object_as_numpy(pickle_object_path):
	with open(pickle_object_path,"rb") as fp:
		picke_data = pickle.load(fp)
	return picke_data

def save_output_dict(M,output_path):
	with open(output_path, "wb") as fp:
		pickle.dump(M, fp, protocol=pickle.HIGHEST_PROTOCOL)
	return

def save_output(L,cluster_centres,output_path):
	M = {'L':L,'cluster_centres':cluster_centres}
	with open(output_path, "wb") as fp:
		pickle.dump(M, fp, protocol=pickle.HIGHEST_PROTOCOL)
	return	

def save_to_pickle(data,output_path):
	with open(output_path, "wb") as fp:
		pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
	return		

def save_image(L,output_path,colors):
	if not colors:
		colors = [[255,255,0],[128,255,0],[0,128,255],[255,0,255],[255,0,0],[0,0,0],[255,128,0],[0,255,128],[0,255,255],[127,0,255],[255,0,127],[128,128,128],[0,51,0],[102,0,0],[255,255,255],[204,229,255]] 
	a = [[[] for y in range(0,len(L[0]))] for z in range(0,len(L))]
	for x in range(0,len(L)):
		for y in range(0,len(L[0])):
			a[x][y] = colors[L[x][y]]
	a = np.array(a,dtype='uint8')
	im = Image.fromarray(a)
	im.save(output_path)
	return	


def get_spaced_colors(cluster_number): 
	max_value = 16581375
	interval = int(max_value / cluster_number)
	colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
	return [[int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)] for i in colors]

