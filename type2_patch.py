import numpy as np
import scipy.io as io
import sqlite3

def load_data():
    input_mat = io.loadmat("mldata/Indian_pines_Preprocessed_patch_3.mat")['preprocessed_img']
    target_mat = io.loadmat("mldata/Indian_pines_Preprocessed_patch_3.mat")['preprocessed_gt']
    target_mat = np.asarray(target_mat, dtype=np.int32)
    return input_mat, target_mat

def get_type_2_patch(target_mat, percnt_pos, start_patch_size, end_patch_size, pos_class):
    class_pixels = np.where(target_mat == pos_class)
    patch_points = [-1, -1, -1, -1]
    n_patch_pos_pixels = (len(class_pixels[0]) * percnt_pos) // 100
    if(n_patch_pos_pixels > 0):
        for k in range(start_patch_size, end_patch_size + 1):
            for i in range(target_mat.shape[0] - k):
                for j in range(target_mat.shape[1] - k):
                    end_row = i + k -1
                    end_col = j + k -1
                    if end_row > target_mat.shape[0] - 1:
                        end_row = target_mat.shape[0] - 1
                    if end_col > target_mat.shape[1] - 1:
                        end_col = target_mat.shape[1] - 1

                    my_patch = target_mat[i:end_row + 1, :][:, j:end_col + 1]
                    class_pixels = np.where(my_patch == pos_class)
                    if len(class_pixels[0]) >= n_patch_pos_pixels:
                        patch_points = [i, end_row, j, end_col]
                        return patch_points
    return patch_points

def save_in_db(query, values):
    conn = sqlite3.connect('nnPU.db')
    c = conn.cursor()
    c.executemany(query, values)
    conn.commit()
    conn.close()

def get_patch_by_class(target_mat):
    n_class = np.max(target_mat) + 1
    # print(n_class)
    start_patch_size = int(target_mat.shape[0] * 0.05)
    end_patch_size  = int(target_mat.shape[0] * 0.25)
    percnt_pos = 20
    for i in range(n_class):
        patch_points = get_type_2_patch(target_mat, percnt_pos, start_patch_size, end_patch_size, i)
        query = '''INSERT INTO PatchClass (class, patch_row_start, patch_row_end, patch_col_start, patch_col_end) VALUES (?, ?, ?, ?, ?) '''
        values = [(i, patch_points[0], patch_points[1], patch_points[2], patch_points[3])]
        save_in_db(query, values)

if __name__ == '__main__':
    input_mat, target_mat = load_data()
    get_patch_by_class(target_mat)