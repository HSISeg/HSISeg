import numpy as np
import utils, Config

def get_patch_by_class(labelled_img, percnt_pos, start_patch_size, end_patch_size, pos_class):
    class_pixels = np.where(labelled_img == pos_class)
    patch_points = [-1, -1, -1, -1]
    n_patch_pos_pixels = (len(class_pixels[0]) * percnt_pos) // 100
    if(n_patch_pos_pixels > 0):
        for k in range(start_patch_size, end_patch_size + 1):
            for i in range(labelled_img.shape[0] - k):
                for j in range(labelled_img.shape[1] - k):
                    end_row = i + k -1
                    end_col = j + k -1
                    if end_row > labelled_img.shape[0] - 1:
                        end_row = labelled_img.shape[0] - 1
                    if end_col > labelled_img.shape[1] - 1:
                        end_col = labelled_img.shape[1] - 1

                    my_patch = labelled_img[i:end_row + 1, :][:, j:end_col + 1]
                    class_pixels = np.where(my_patch == pos_class)
                    if len(class_pixels[0]) >= n_patch_pos_pixels:
                        patch_points = [i, end_row, j, end_col]
                        return patch_points
    return patch_points



def gen_and_save_patch(labelled_img):
    n_class = np.max(labelled_img) + 1
    start_patch_size = int(labelled_img.shape[0] * Config.patch_window_start_percent) // 100
    end_patch_size  = int(labelled_img.shape[0] * Config.patch_window_end_percent) // 100

    for i in range(n_class):
        patch_points = get_patch_by_class(labelled_img, Config.percnt_pos, start_patch_size, end_patch_size, i)
        query = '''INSERT INTO PatchClass (class, patch_row_start, patch_row_end, patch_col_start, patch_col_end) VALUES (?, ?, ?, ?, ?) '''
        values = [(i, patch_points[0], patch_points[1], patch_points[2], patch_points[3])]
        utils.save_in_db(query, values)

if __name__ == '__main__':
    data_img, labelled_img = utils.load_preprocessed_data()
    gen_and_save_patch(labelled_img)