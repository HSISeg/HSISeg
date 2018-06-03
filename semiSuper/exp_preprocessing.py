import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "HSISeg.settings")
django.setup()
import numpy as np
import Config, importlib
from algo import image_helper as ih
import preprocess_data
from algo import centroid_init
from algo import algo_default_params as cluster_default_params
from algo import pdhg_weight

PATCH_SIZE = Config.PATCH_SIZE
data = Config.data

def get_preprocessed_data():
    input_mat, target_mat = preprocess_data.download_and_save_data()
    image_pickle_file_path = Config.out + "_" + data + "_image_data.pickle"
    ih.save_to_pickle(input_mat, image_pickle_file_path)
    preprocessed_pickle_file_path = Config.out + data + "_Preprocessed_patch_" + str(PATCH_SIZE) + ".pickle"
    try:
        preprocessed_data = ih.get_pickle_object_as_numpy(preprocessed_pickle_file_path)
        preprocessed_img = preprocessed_data["preprocessed_img"]
        preprocessed_gt = preprocessed_data["preprocessed_gt"]
    except Exception as e:
        print(e, 'preprocess data')
        preprocessed_img = preprocess_data.preprocess_data(input_mat, target_mat)
        preprocessed_data = {}
        preprocessed_img = np.asarray(preprocessed_img, dtype=np.float32)
        preprocessed_gt = np.asarray(target_mat, dtype=np.int32)
        print("saving data ....")
        preprocessed_data["preprocessed_img"] = preprocessed_img
        preprocessed_data["preprocessed_gt"] = preprocessed_gt
        ih.save_to_pickle(preprocessed_data, preprocessed_pickle_file_path)

    # clustering
    centroid_pickle_file_path = Config.out + data + "_initial_centroid.pickle"
    maxconn = cluster_default_params.gen_default_params['maxconn']
    image = ih.get_pickle_object_as_numpy(image_pickle_file_path)
    cluster_number = cluster_default_params.cluster_number
    weight_pickle_file_path = Config.out + data + "_weight.pickle"
    output_path = Config.out + data + "_output"
    cluster_algo = cluster_default_params.default_cluster_algo
    params = cluster_default_params.algo_details[cluster_algo]['algo_params']
    cluster_img_pickle_file_path = output_path + "_data_" + str(params['outerloop']) + ".pickle"
    try:
        centroid = ih.get_pickle_object_as_numpy(centroid_pickle_file_path)
    except:
        centroid_algo = cluster_default_params.default_centroid_init_algo
        centroid_algo_func = getattr(centroid_init, centroid_algo)
        centroid = centroid_algo_func(image, cluster_number)
        ih.save_to_pickle(centroid, centroid_pickle_file_path)

    try:
        weight = ih.get_pickle_object_as_numpy(weight_pickle_file_path)
    except:
        weight_algo = cluster_default_params.default_weight_algo
        weight_algo_params = cluster_default_params.weight_algo[weight_algo]['algo_params']
        weight_algo_func = getattr(pdhg_weight, weight_algo)
        weight = weight_algo_func(image, weight_algo_params)
        ih.save_to_pickle(weight, weight_pickle_file_path)

    try:
        res = ih.get_pickle_object_as_numpy(cluster_img_pickle_file_path)
        clust_labelled_img = res['L']
        # clust_prob_labelled_img = res['U']
    except:
        params['weight_pickle_file_path'] = weight_pickle_file_path
        params['centroid_pickle_file_path'] = centroid_pickle_file_path
        file = importlib.import_module(cluster_default_params.algo_main_func[cluster_algo]['file'])
        main_func = getattr(file, cluster_default_params.algo_main_func[cluster_algo]['func'])
        main_func(image, cluster_number, output_path, maxconn, params, None)
        res = ih.get_pickle_object_as_numpy(cluster_img_pickle_file_path)
        clust_labelled_img = res['L']
        # clust_prob_labelled_img = res['U']
    clust_prob_labelled_img = None
    clust_labelled_img = np.asarray(clust_labelled_img, dtype=np.int32)
    return clust_labelled_img, clust_prob_labelled_img, preprocessed_img, target_mat