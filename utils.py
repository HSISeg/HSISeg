import chainer
from chainer import cuda
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from chainer import functions as F
import numpy as np
import Config
import scipy.io as io
import sqlite3, pickle
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
from algo.models import PNstats, PUstats

def get_threshold(model, X, Y):
    xp = cuda.get_array_module(X, False)
    if Config.output_layer_activation not in ['sigmoid', 'sign']:
        raise ValueError("Only support sigmoid or sign for last layer activation.")
    size = X.shape[0]
    with chainer.no_backprop_mode():
        with chainer.using_config("train", False):
            if Config.output_layer_activation == 'sigmoid':
                h = xp.reshape(F.sigmoid(model.calculate(X)).data, size)
            else:
                h = xp.reshape(xp.sign(model.calculate(X).data), size)

    if isinstance(h, chainer.Variable):
        h = h.data
    fpr, tpr, threshold = roc_curve(Y, h, pos_label=1)
    model.auc = roc_auc_score(Y, h)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]
    model.threshold = list(roc_t['threshold'])[0]
    return model.threshold

def get_output_by_activation(model, x):
    xp = cuda.get_array_module(x, False)
    if Config.output_layer_activation not in ['sigmoid','sign']:
        raise ValueError("Only support sigmoid or sign for last layer activation.")
    size = x.shape[0]
    with chainer.no_backprop_mode():
        with chainer.using_config("train", False):
            if Config.output_layer_activation == 'sigmoid':
                h = xp.reshape(F.sigmoid(model.calculate(x)).data, size)
            else:
                h = xp.reshape(xp.sign(model.calculate(x).data), size)
    if isinstance(h, chainer.Variable):
        h = h.data
    if Config.output_layer_activation == 'sigmoid':
        h[np.where(h >= model.threshold)] = 1 # For binary
        h[np.where(h < model.threshold)] = 0 # For binary

    return h

def set_model_auc(model, predicted_output, true_output):
    model.auc = roc_auc_score(true_output, predicted_output)
    return

def get_model_stats(predicted_output, true_output):
    if isinstance(predicted_output, chainer.Variable):
        predicted_output = predicted_output.data
    if isinstance(true_output, chainer.Variable):
        true_output = true_output.data
    try:
        precision, recall, _, _ = precision_recall_fscore_support(true_output, predicted_output, pos_label = 1, average='binary')
    except:
        precision, recall = 0.0, 0.0
    try:
        tn, fp, fn, tp = confusion_matrix(true_output, predicted_output).ravel()
    except:
        tn, tp, fp, fn = 0, 0, 0, 0
    return precision, recall, (int(tn), int(fp), int(fn), int(tp))

def shuffle_data(X, Y):
    perm = np.random.permutation(len(Y))
    X, Y = X[perm], Y[perm]
    return X, Y

def load_preprocessed_data():
    data_img = io.loadmat("mldata/" + Config.data + "_Preprocessed_patch_3.mat")['preprocessed_img']
    labelled_img = io.loadmat("mldata/" + Config.data + "_Preprocessed_patch_3.mat")['preprocessed_gt']
    labelled_img = np.asarray(labelled_img, dtype=np.int32)
    data_img = np.asarray(data_img, dtype=np.float32)
    return data_img, labelled_img

def load_clustered_img():
    with open("mldata/" + Config.data + "_clustered_img.pickle", "rb") as fp:
        pickle_data = pickle.load(fp, encoding='latin-1')
    clust_labelled_img = np.asarray(pickle_data['L'], dtype=np.int32)
    return clust_labelled_img

def save_pickle(data, file):
    with open(file, "wb") as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return

def load_sampled_data(file):
    with open(file, "rb") as fp:
        pickle_data = pickle.load(fp)
    XYtrain, XYtest, prior, testX, testY, trainX, trainY, crossX, crossY = pickle_data["XYtrain"], pickle_data["XYtest"], pickle_data["prior"], pickle_data["testX"], pickle_data["testY"], pickle_data["trainX"], pickle_data["trainY"], pickle_data["crossX"], pickle_data["crossY"]
    train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, shuffled_test_pixels = pickle_data["train_lp_pixels"], pickle_data["train_up_pixels"], pickle_data["train_un_pixels"], pickle_data["test_pos_pixels"], pickle_data["test_neg_pixels"], pickle_data["shuffled_test_pixels"]
    return XYtrain, XYtest, prior, testX, testY, trainX, trainY, crossX, crossY, train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, shuffled_test_pixels


def save_data_in_PUstats(values):
    conn = sqlite3.connect('nnPU.db')
    c = conn.cursor()
    query = '''INSERT INTO PUstats (pos_class, neg_class, precision, recall, true_pos, true_neg, false_pos, false_neg, test_type, exclude_class_indx, no_train_pos_labelled, no_train_pos_unlabelled, no_train_neg_unlabelled, visual_result_filename, train_pos_neg_ratio, threshold, auc) VALUES (?, ?, ?, ?, ?, ?, ?, ? ,?, ?, ?, ?, ?, ?, ?, ?, ?) '''
    c.executemany(query, [values])
    conn.commit()
    conn.close()

def save_in_db(query, values):
    conn = sqlite3.connect('nnPU.db')
    c = conn.cursor()
    c.executemany(query, values)
    conn.commit()
    conn.close()


def get_patch_points_by_class(class_indx):
    conn = sqlite3.connect('nnPU.db')
    c = conn.cursor()
    query = "SELECT patch_row_start, patch_row_end, patch_col_start, patch_col_end from PatchClass where class = ? order by creation_time desc limit 1"
    values = (class_indx,)
    c.execute(query, values)
    rows = c.fetchall()
    result = None
    conn.commit()
    conn.close()
    if len(rows) > 0 and rows[0][0] != -1:
        result = rows[0]
    return result

def get_indices_from_list(labelled_img, indices_list):
    indx = np.isin(labelled_img, indices_list)
    return indx

def check_if_test_done(pos_class, test_type, neg_class):
    conn = sqlite3.connect('nnPU.db')
    c = conn.cursor()
    query = '''SELECT * FROM PUstats WHERE pos_class = ? and neg_class = ? and test_type = ? '''
    c.execute(query, (str(pos_class), neg_class, str(test_type)))
    rows = c.fetchall()
    if len(rows) > 0:
        return True
    conn.commit()
    conn.close()
    return False

def check_if_test_done_models(pos_class, test_type, neg_class, data_name, train_pos_neg_ratio, isPU):
    if isPU:
        rows = PUstats.objects.filter(pos_class = pos_class, neg_class = neg_class, data_name= data_name, train_pos_neg_ratio = train_pos_neg_ratio, test_type = test_type)
    else:
        rows = PNstats.objects.filter(pos_class = pos_class, neg_class = neg_class, data_name= data_name, train_pos_neg_ratio = train_pos_neg_ratio, test_type = test_type)
    if rows:
        return True
    return False

def get_excluded_pixels(labelled_img, train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels):
    selected_indx = np.zeros(labelled_img.shape, dtype=np.bool)
    selected_indx[train_lp_pixels] = True
    selected_indx[train_up_pixels] = True
    selected_indx[train_un_pixels] = True
    selected_indx[test_pos_pixels] = True
    selected_indx[test_neg_pixels] = True
    exclude_pixels = np.where(selected_indx == False)
    return exclude_pixels

def get_binary_gt_img(labelled_img, train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, exclude_pixels):
    gt_img = np.array(labelled_img, copy=True)
    gt_img[exclude_pixels] = -1
    gt_img[train_up_pixels] = 1
    gt_img[train_lp_pixels] = 1
    gt_img[train_un_pixels] = 0
    gt_img[test_pos_pixels] = 1
    gt_img[test_neg_pixels] = 0
    return gt_img

def get_binary_data(pos_pixels, neg_pixels, data_img):
    X = np.concatenate((data_img[pos_pixels], data_img[neg_pixels]))
    Y = np.asarray(np.concatenate((np.full(len(pos_pixels[0]), 1), np.full(len(neg_pixels[0]), 0))), dtype=np.int32)
    return X, Y

def get_binary_predicted_image(labelled_img, model, test_X, train_lp_pixels, train_up_pixels, train_un_pixels, shuffled_test_pixels, exclude_pixels):
    predicted_img = np.array(labelled_img, copy=True)
    predicted_img[exclude_pixels] = -1
    predicted_img[train_up_pixels] = -1
    predicted_img[train_lp_pixels] = -1
    predicted_img[train_un_pixels] = -1
    predicted_output = get_output_by_activation(model, test_X)
    predicted_img[shuffled_test_pixels] = predicted_output
    return predicted_img, predicted_output

#
# def gen_visual_results_data(target_mat, model, input_mat, train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels ):
#     selected_indx = np.zeros(target_mat.shape, dtype=np.bool)
#     selected_indx[train_lp_pos_pixels] = True
#     selected_indx[train_up_pos_pixels] =  True
#     selected_indx[train_neg_pixels] = True
#     selected_indx[test_pos_pixels] =True
#     selected_indx[test_neg_pixels] = True
#     exclude_pixels = np.where(selected_indx == False)
#     # print(target_mat.dtype, input_mat.dtype)
#     gt_img = np.array(target_mat, copy=True)
#     gt_img[exclude_pixels] = -1
#     gt_img[train_up_pos_pixels] = 1
#     gt_img[train_lp_pos_pixels] = 1
#     gt_img[test_pos_pixels] = 1
#     gt_img[train_neg_pixels] = 0
#     gt_img[test_neg_pixels] = 0
#     predicted_img = np.array(target_mat, copy=True)
#     predicted_img[exclude_pixels] = -1
#     predicted_img[train_up_pos_pixels] = -1
#     predicted_img[train_lp_pos_pixels] = -1
#     x = np.concatenate((input_mat[test_pos_pixels], input_mat[test_neg_pixels]), axis=0)
#     # print(x.shape)
#     t = np.concatenate((np.full(len(test_pos_pixels[0]), 1, dtype=np.int32), np.full(len(test_neg_pixels[0]), 0, dtype=np.int32)), axis=0)
#     test_pixels = (np.concatenate((test_pos_pixels[0], test_neg_pixels[0]), axis=0), np.concatenate((test_pos_pixels[1], test_neg_pixels[1]), axis=0))
#     precision, recall, (tn, fp, fn, tp), h = get_accuracy(model, x, t)
#     predicted_img[test_pixels] = h
#     return gt_img, predicted_img, train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels, exclude_pixels, (precision, recall, int(tp), int(tn), int(fp), int(fn) )
