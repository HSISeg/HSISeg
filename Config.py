# class A:
# 	data = None
# 	def __init__(self, data):
# 		self.data = data

# opt = A("Indian_pines")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Indian_pines')
opt = parser.parse_args()
import numpy as np 
data_based_config = {"Indian_pines":{"data_key": "indian_pines", "nbands": 10, "url1": "http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat",
                                     "url2": "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat", "include_class_list" :[2, 3, 5, 6, 8, 10, 11, 12, 14],
                                     "pos_class_list":[2]},
                     "Salinas": {"data_key": "salinas", "nbands": 8, "url1": "http://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat",
                                     "url2": "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat", "include_class_list" : [6, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                     "pos_class_list":[6]},
                    "PaviaU": {"data_key": "paviaU", "nbands": 8, "url1": "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
                                     "url2": "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat", "include_class_list" : [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     "pos_class_list":[1]},
                    }
#NNPU config
batchsize = 100
epoch = 2
loss = 'sigmoid_cross_entropy'
model = 'bass_net'
gamma = 1.
beta = 0.
stepsize = 1e-3
out = 'mldata/'
output_layer_activation = 'sigmoid'
epsilon = 10**-4
# Zero-origin GPU ID (negative value indicates CPU)
gpu = -1
unlabeled_tag = 0
default_positive_prior = 0.33
PATCH_SIZE = 3
nbands = data_based_config[opt.data]["nbands"]
# Data name Can be Indian_pines, Salinas, PaviaU
data = opt.data
data_key = data_based_config[opt.data]["data_key"]
url1 = data_based_config[opt.data]["url1"]
url2 = data_based_config[opt.data]["url2"]

"""test type 1 config"""
type_1_train_pos_labelled_percentage = 60

# percentage of pixels from total pixels of positive class that will be included in training, type_1_train_pos_labelled_percentage% of positive pixels will be labelled
type_1_train_pos_percentage = 10
temperature_test_list = [16]
# np.arange(14,50) 
# temperature_test_list = [20]
# baseline_test_list = [30]
baseline_test_list =  [18]
# np.arange(20,50) 
# percentage of pixels from total pixels of negative class (include_class_list - positive_class) that will be included in training
# type_1_train_neg_percentage = 30 # not required

type_1_neg_pos_ratio_in_train = [1, 0.82, 0.67, 0.54, 0.43, 0.33, 0.25, 0.18]
# type_1_pos_neg_ratio_in_train = [0.33]
# type_1_pos_neg_ratio_in_train = [1]
type_1_cross_pos_percentage = 7
# type_1_pos_neg_ratio_in_cross = 1
experiment_number = 20
is_random_positive_sampling = False


type_1_include_class_list = data_based_config[opt.data]["include_class_list"]
pos_class_list = data_based_config[opt.data]["pos_class_list"]





""" 
boolean value to indicate the way of selection for negative unlabelled training data,
If set to true then all the data from negative class list are accumulated and randomly train_neg_percentage of the total data is selected
In the above case it might happen that training data might not have any pixel for class k in negative class list 
If set to false then train_neg_percentage pixels of data is randomly selected for each class in negative class list
The above ensures that training data has some pixels of each class in negative class list 
"""
# is_random_neg = True

# class label included in this test, negative class  = list(set(include_class_list) - set(positive_class))
# Note: Class labels should be present in groundtooth image
# type_1_include_class_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# type_1_include_class_list = [2, 3, 5, 6, 8, 10, 11, 12, 14]

# type_1_include_class_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # [2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # [2, 3, 5, 6, 8, 10, 11, 12, 14]
# type_1_include_class_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

"""Test type 2 config"""
# Minimum percentage of total positive pixels that should be present in the patch
percnt_pos = 30
patch_window_start_percent = 5
patch_window_end_percent = 25

# class label included in this test, negative class  = list(set(include_class_list) - set(positive_class))
# Note: Class labels should be present in groundtooth image
# include_class_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# type_2_include_class_list = [2, 3, 5, 6, 8, 10, 11, 12, 14]

