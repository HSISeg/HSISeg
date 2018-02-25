import numpy as np
import chainer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from chainer import Variable, functions as F
import sqlite3

def get_accuracy(model,x,t):
    size = len(t)
    with chainer.no_backprop_mode():
        with chainer.using_config("train", False):
            h = np.reshape(F.sigmoid(model.calculate(x)).data, size) # For binary
    if isinstance(h, chainer.Variable):
        h = h.data

    h[np.where(h >= 0.5)] = 1 # For binary
    h[np.where(h < 0.5)] = 0 # For binary

    if isinstance(t, chainer.Variable):
        t = t.data
    try:
        precision, recall, _, _ = precision_recall_fscore_support(t, h, pos_label = 1, average='binary')
    except:
        precision, recall = 0.0, 0.0

    tn, fp, fn, tp = confusion_matrix(t, h).ravel()
    return precision, recall, (tn, fp, fn, tp), h

def save_data_in_PUstats(values):
    conn = sqlite3.connect('nnPU.db')
    c = conn.cursor()
    query = '''INSERT INTO PUstats (pos_class, neg_class, precision, recall, true_pos, true_neg, false_pos, false_neg, test_type, exclude_class_indx, no_train_pos_labelled, no_train_pos_unlabelled, no_train_neg_unlabelled, visual_result_filename) VALUES (?, ?, ?, ?, ?, ?, ?, ? ,?, ?, ?, ?, ?, ?) '''
    c.executemany(query, [values])
    conn.commit()
    conn.close()

def check_if_test_done(pos_label, test_type, neg_labels_list):
    conn = sqlite3.connect('nnPU.db')
    c = conn.cursor()
    query = '''SELECT * FROM PUstats WHERE pos_class = ? and neg_class = ? and test_type = ? '''
    c.execute(query, (str(pos_label), ",".join([str(i) for i in neg_labels_list]), str(test_type)))
    rows = c.fetchall()
    if len(rows) > 0:
        return True
    conn.commit()
    conn.close()
    return False

def gen_visual_results_data(target_mat, model, input_mat, train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels ):
    selected_indx = np.zeros(target_mat.shape, dtype=np.bool)
    selected_indx[train_lp_pos_pixels] = True
    selected_indx[train_up_pos_pixels] =  True
    selected_indx[train_neg_pixels] = True
    selected_indx[test_pos_pixels] =True
    selected_indx[test_neg_pixels] = True
    exclude_pixels = np.where(selected_indx == False)
    # print(target_mat.dtype, input_mat.dtype)
    gt_img = np.array(target_mat, copy=True)
    gt_img[exclude_pixels] = -1
    gt_img[train_up_pos_pixels] = 1
    gt_img[train_lp_pos_pixels] = 1
    gt_img[test_pos_pixels] = 1
    gt_img[train_neg_pixels] = 0
    gt_img[test_neg_pixels] = 0
    predicted_img = np.array(target_mat, copy=True)
    predicted_img[exclude_pixels] = -1
    predicted_img[train_up_pos_pixels] = -1
    predicted_img[train_lp_pos_pixels] = -1
    x = np.concatenate((input_mat[test_pos_pixels], input_mat[test_neg_pixels]), axis=0)
    # print(x.shape)
    t = np.concatenate((np.full(len(test_pos_pixels[0]), 1, dtype=np.int32), np.full(len(test_neg_pixels[0]), 0, dtype=np.int32)), axis=0)
    test_pixels = (np.concatenate((test_pos_pixels[0], test_neg_pixels[0]), axis=0), np.concatenate((test_pos_pixels[1], test_neg_pixels[1]), axis=0))
    precision, recall, (tn, fp, fn, tp), h = get_accuracy(model, x, t)
    predicted_img[test_pixels] = h
    return gt_img, predicted_img, train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels, exclude_pixels, (precision, recall, int(tp), int(tn), int(fp), int(fn) )
