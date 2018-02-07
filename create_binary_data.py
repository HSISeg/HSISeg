import numpy as np
import scipy.io

def get_pos_neg_data():
    mat = scipy.io.loadmat("mldata/Indian_pines_Full_Train_patch_3.mat")
    X_tr = mat["train_patch"]
    Y_tr = mat["train_labels"]
    mat = scipy.io.loadmat("mldata/Indian_pines_Test_patch_3.mat")
    X_te = mat["test_patch"]
    Y_te = mat["test_labels"]
    Y_tr = Y_tr.astype(np.int32)
    X_tr = X_tr.astype(np.float32)
    X_tr = X_tr[0]
    Y_tr = np.reshape(Y_tr, (Y_tr.shape[0] * Y_tr.shape[1]))
    X_te = X_te.astype(np.float32)
    Y_te = Y_te.astype(np.int32)
    Y_te = np.reshape(Y_te, (Y_te.shape[0] * Y_te.shape[1]))
    class_list = np.unique(Y_tr)
    pos_class_indx = class_list[np.random.randint(0,len(class_list))]
    # pos_class_indx = 0
    pos_class = class_list[pos_class_indx]
    class_list[-1], class_list[pos_class_indx] = class_list[pos_class_indx],class_list[-1]
    neg_class_indx = class_list[np.random.randint(0, len(class_list) - 1)]
    # neg_class_indx = 1
    neg_class = class_list[neg_class_indx]
    print("positive labelled class",pos_class, "negative labelled class", neg_class)
    tr_pos_indx = np.where(Y_tr == pos_class)
    tr_neg_indx = np.where(Y_tr == neg_class)
    X_tr_pos = X_tr[tr_pos_indx]
    Y_tr_pos = Y_tr[tr_pos_indx]
    X_tr_neg = X_tr[tr_neg_indx]
    Y_tr_neg = Y_tr[tr_neg_indx]
    print("training positive data count", len(tr_pos_indx[0]), "training negative data count", len(tr_neg_indx[0]))
    X_tr = np.concatenate((X_tr_pos, X_tr_neg), axis=0)
    Y_tr = np.concatenate((Y_tr_pos, Y_tr_neg), axis=0)
    te_pos_indx = np.where(Y_te == pos_class)
    te_neg_indx = np.where(Y_te == neg_class)
    print("testing positive data count", len(te_pos_indx[0]), "training negative data count", len(te_neg_indx[0]))
    X_te_pos = X_te[te_pos_indx]
    Y_te_pos = Y_te[te_pos_indx]
    X_te_neg = X_te[te_neg_indx]
    Y_te_neg = Y_te[te_neg_indx]
    X_te = np.concatenate((X_te_pos, X_te_neg), axis=0)
    Y_te = np.concatenate((Y_te_pos, Y_te_neg), axis=0)
    print(np.unique(Y_tr), "Y_tr_labels", np.unique(Y_te), "Y_te_labels")
    return (X_tr, Y_tr), (X_te, Y_te), pos_class

def binarize_indian_pines(labels, class_indx):
    labels[np.where(labels != class_indx)] = -1
    labels[np.where(labels != -1)] = 1
    labels[np.where(labels == -1)] = 0
    return labels

(X_tr, Y_tr), (X_te, Y_te), pos_class = get_pos_neg_data()
Y_tr = binarize_indian_pines(Y_tr, pos_class)
Y_te = binarize_indian_pines(Y_te, pos_class)
patch_size = X_tr.shape[2]
full_train = {}
full_train["train_patch"] = X_tr
full_train["train_labels"] = Y_tr
scipy.io.savemat("mldata/Indian_pines_Binary_Full_Train_patch_" + str(patch_size) + ".mat", full_train)
test = {}
test["test_patch"] = X_te
test["test_labels"] = Y_te
scipy.io.savemat("mldata/Indian_pines_Binary_Test_patch_" + str(patch_size) + ".mat", test)

