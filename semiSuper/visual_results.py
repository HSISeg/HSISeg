import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
import pickle


def get_color_list():
	return {
		"black": [0,0,0],
		"white": [255,255,255],
		"red": [255,0,0],
		"lime": [0,255,0],
		"blue": [0,0,255],
		"yellow": [255,255,0],
		"cyan": [0,255,255],
		"magenta": [255,0,255],
		"silver": [192,192,192],
		"gray": [128,128,128],
		"maroon": [128,0,0],
		"olive": [128,128,0],
		"green": [0,128,0],
		"purple": [128,0,128],
		"teal": [0,128,128],
		"navy": [0,0,128]
	}

def get_color_to_legend():
	return {
		"lime" : "label pos / test data pos / accurate",
		"red" : "unlabel neg / test negative / inaccurate",
		"yellow" : "unlabel pos",
		"black" : "exclude",
		"green" : "true positive",
		"maroon" : "false positive",
		"olive" : "true negative",
		"purple" : "false negative"
	}


def generate_and_save_visualizations(ground_truth_image,predicted_image, train_lp_pixels, train_up_pixels,\
	train_un_pixels, test_p_pixels, test_n_pixels, exclude_pixels, output_path):
	train_lp_pixels = numpy_to_tuple_array(train_lp_pixels)
	train_up_pixels = numpy_to_tuple_array(train_up_pixels)
	train_un_pixels = numpy_to_tuple_array(train_un_pixels)
	test_p_pixels = numpy_to_tuple_array(test_p_pixels)
	test_n_pixels = numpy_to_tuple_array(test_n_pixels)
	exclude_pixels = numpy_to_tuple_array(exclude_pixels)
	## first we will generate all the images -- then we will print them on matplotlib

	## compute the confusion matrix ###########
	if ground_truth_image.shape != predicted_image.shape:
		print ("size missmatch in ground truth and predicted image")
		raise Exception("size missmatch in ground truth and predicted image")

	image_size = ground_truth_image.shape

	m_dim = image_size[0]
	n_dim = image_size[1]

	color_list = get_color_list()

	## the experiment setting ##
	experiment_setting = np.full((m_dim, n_dim , 3) , 0 , 'uint8' )
	# print(train_lp_pixels)
	for (m,n) in train_lp_pixels:
		experiment_setting[m][n] = color_list["lime"]

	for (m,n) in train_un_pixels:
		experiment_setting[m][n] = color_list["red"]

	for (m,n) in train_up_pixels:
		experiment_setting[m][n] = color_list["yellow"]


	## making ground truth and predicted images ##
	ground_truth_color = np.full((m_dim, n_dim , 3) , 0 , 'uint8' )
	predicted_color = np.full((m_dim, n_dim , 3) , 0 , 'uint8' )

	for (m,n) in test_p_pixels + test_n_pixels :
		if ground_truth_image[m][n] == 1:
			ground_truth_color[m][n] = color_list["lime"]
		if ground_truth_image[m][n] == 0:
			ground_truth_color[m][n] = color_list["red"]
		if predicted_image[m][n] == 1:
			predicted_color[m][n] = color_list["lime"]
		if predicted_image[m][n] == 0:
			predicted_color[m][n] = color_list["red"]

	for (m,n) in exclude_pixels:
		ground_truth_color[m][n] = color_list["black"]
		predicted_color[m][n] = color_list["black"]
	

	## confusion image ##
	total_test = test_p_pixels + test_n_pixels
	true_positive_count = 0
	false_positive_count = 0
	true_negative_count = 0
	false_negative_count = 0
	confusion = np.full((m_dim, n_dim , 3) , 0 , dtype = 'uint8')
	for (m,n) in total_test:
		if predicted_image[m][n] == 1:
			if predicted_image[m][n] == ground_truth_image[m][n]:
				## true positive
				confusion[m][n] = color_list["green"]
				true_positive_count += 1
			else:
				## false positive
				confusion[m][n] = color_list["maroon"]
				false_positive_count += 1
		else:
			if predicted_image[m][n] == ground_truth_image[m][n]:
				## true negative
				confusion[m][n] = color_list["olive"]
				true_negative_count += 1
			else:
				## false negative
				confusion[m][n] = color_list["purple"]
				false_negative_count += 1

	## accuracy image ##
	accuracy = np.full((m_dim, n_dim, 3) , 0 , dtype = 'uint8')
	accurate_count = 0
	inaccurate_count = 0
	for (m,n) in test_p_pixels + test_n_pixels:
		if predicted_image[m][n] == ground_truth_image[m][n]:
			## accurate
			accuracy[m][n] = color_list["lime"]
			accurate_count += 1
		else:
			## inaccurate
			accuracy[m][n] = color_list["red"]
			inaccurate_count += 1

	clust_data = np.random.random((10,3))
	collabel=("col 1", "col 2", "col 3")

	try:
		precision = float(true_positive_count) / float(true_positive_count + false_positive_count)
	except Exception as e:
		precision = -1

	try:
		recall = float(true_positive_count) / float(true_positive_count + false_negative_count)
	except Exception as e:
		recall = -1

	total_train_count = len(train_lp_pixels) + len(train_up_pixels) + len(train_un_pixels)
	clust_data = [
		["total_pixel",	m_dim * n_dim,				"tp",			float(true_positive_count) *100.0 / float(len(total_test))							],
		["total_lp", 	len(train_lp_pixels),		"fp",			float(false_positive_count) *100.0 / float(len(total_test))							],
		["total_up", 	len(train_up_pixels),		"tn",			float(true_negative_count) *100.0 / float(len(total_test))							],
		["total_un", 	len(train_un_pixels),		"fn",			float(false_negative_count) *100.0 / float(len(total_test))							],
		["total_train", total_train_count,			"accurate",		float(accurate_count) *100.0 / float(len(total_test))								],
		["test_p", 		len(test_p_pixels),			"inaccurace",	float(inaccurate_count) *100.0 / float(len(total_test))								],
		["test_n", 		len(test_n_pixels),			"precision",	precision																			],
		["exclude",		len(exclude_pixels),		"recall",		recall																				]
	]



	image_grid = [[(experiment_setting,"Experiment Settings","image"), (ground_truth_color, "TEST ground truth","image"), (predicted_color, "TEST predicted","image") ],\
		[(confusion, "confusion","image"),(accuracy, "accuracy","image"),((clust_data,None),"statistics","table")]]


	create_and_save_plot(image_grid, get_color_to_legend(), color_list, output_path)
	return


def create_and_save_plot(image_grid, legend_list, color_list, output_path):
	# image grid is assumed to be 2 dimention array of images given in the same order
	# to be displayed on the plot
	m_size = len(image_grid)
	n_size = max(map(lambda x:len(x), image_grid))
	plt.close('all')

	f, axarr = plt.subplots(m_size, n_size)

	m = 0
	for plot_line in image_grid:
		n = 0
		for plot in plot_line:
			if plot[2] == "image":
				axarr[m][n].imshow(plot[0])
				axarr[m][n].set_title(plot[1])
			elif plot[2] == "table":
				axarr[m][n].axis('tight')
				axarr[m][n].axis('off')
				axarr[m][n].table(cellText=plot[0][0],colLabels=plot[0][1],loc='center')
				axarr[m][n].set_title(plot[1])
			else:
				f.delaxes(axarr[m][n])
			n += 1
		for i in range(n,n_size):
			f.delaxes(axarr[m][i])
		m += 1


	patches = [mpatches.Patch(color = list(map(lambda x:float(x) / 255.0, color_list[color_name])), label = legend_list[color_name]) for color_name in legend_list]

	# f.legend(handles=patches, bbox_to_anchor=(0., 1.02, 1., .102), loc=4, mode="expand", borderaxespad=0.)
	f.legend(handles=patches, loc=4, bbox_to_anchor=(1.5, 0.5) , borderaxespad=0.0)

	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=.1)
	plt.show()
	plt.savefig(output_path, bbox_inches='tight', dpi=1200)
	return

def extract_random_pixels(total_pixel, size):
	selection_set = set(random.sample(total_pixel, size))
	total_pixel = total_pixel - selection_set
	return list(selection_set),total_pixel


def numpy_to_tuple_array(numpy_list):
	return list(zip(numpy_list[0],numpy_list[1]))

def test_from_pickle():
	file_name = "result/type_2_test_12_pos.pickle"
	with open(file_name,"rb") as fp:
		data = pickle.load( fp )

	# dict_keys(['gt_img', 'predicted_img', 'train_lp_pos_pixels', 'train_up_pos_pixels', 'train_un_pixels', 'test_pos_pixels', 'test_neg_pixels', 'exclude_pixels'])

	ground_truth_image = data["gt_img"]
	predicted_image = data["predicted_img"]
	train_lp_pixels = data["train_lp_pos_pixels"]
	train_up_pixels = data["train_up_pos_pixels"]
	train_un_pixels = data["train_un_pixels"]
	test_p_pixels = data["test_pos_pixels"]
	test_n_pixels = data["test_neg_pixels"]
	exclude_pixels = data["exclude_pixels"]

	output_path = "test_real.png"

	# print (data.keys())

	generate_and_save_visualizations(ground_truth_image,predicted_image, train_lp_pixels, train_up_pixels,\
	train_un_pixels, test_p_pixels, test_n_pixels, exclude_pixels, output_path)

	return

# test_from_pickle()