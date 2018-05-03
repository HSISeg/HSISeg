#NNPU config
batchsize = 100
epoch = 5
loss = 'sigmoid_cross_entropy'
model = 'bass_net'
gamma = 1.
beta = 0.
stepsize = 1e-3
out = 'mldata/'
output_layer_activation = 'sigmoid'
# Zero-origin GPU ID (negative value indicates CPU)
gpu = -1
unlabeled_tag = 0
default_positive_prior = 0.5
PATCH_SIZE = 3
# Data name Can be Indian_pines, Salinas, PaviaU
data = "Indian_pines"
url1 = "http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat"
url2 = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"

"""test type 1 config"""
type_1_train_pos_labelled_percentage = 60
# percentage of pixels from total pixels of positive class that will be included in training, type_1_train_pos_labelled_percentage% of positive pixels will be labelled
type_1_train_pos_percentage = 10
# percentage of pixels from total pixels of negative class (include_class_list - positive_class) that will be included in training
type_1_train_neg_percentage = 30 # not required
type_1_pos_neg_ratio_in_train = [1, 0.82, 0.67, 0.54, 0.43, 0.33, 0.25, 0.18]
type_1_cross_pos_percentage = 7
type_1_pos_neg_ratio_in_cross = 1
""" 
boolean value to indicate the way of selection for negative unlabelled training data,
If set to true then all the data from negative class list are accumulated and randomly train_neg_percentage of the total data is selected
In the above case it might happen that training data might not have any pixel for class k in negative class list 
If set to false then train_neg_percentage pixels of data is randomly selected for each class in negative class list
The above ensures that training data has some pixels of each class in negative class list 
"""
is_random_neg = True

# class label included in this test, negative class  = list(set(include_class_list) - set(positive_class))
# Note: Class labels should be present in groundtooth image
# include_class_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
type_1_include_class_list = [2, 3, 5, 6, 8, 10, 11, 12, 14]
# type_1_include_class_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

"""Test type 2 config"""
# Minimum percentage of total positive pixels that should be present in the patch
percnt_pos = 30
patch_window_start_percent = 5
patch_window_end_percent = 25

# class label included in this test, negative class  = list(set(include_class_list) - set(positive_class))
# Note: Class labels should be present in groundtooth image
# include_class_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
type_2_include_class_list = [2, 3, 5, 6, 8, 10, 11, 12, 14]