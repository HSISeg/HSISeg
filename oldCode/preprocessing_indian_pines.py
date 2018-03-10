import numpy as np
import scipy.ndimage
import scipy.io as io
import os

PATCH_SIZE = 3
data = "Indian_pines"
url1 = "http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat"
url2 = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"

# parser = argparse.ArgumentParser()
# parser.add_argument('--data', type=str, default='Indian_pines')
# parser.add_argument('--patch_size', type=int, default=3)
# opt = parser.parse_args()

def load_data():
    try:
        input_mat = io.loadmat('mldata/' + data + '.mat')[data.lower()]
        target_mat = io.loadmat('mldata/' + data + '_gt.mat')[data.lower() + '_gt']
    except:
        os.system('wget' + ' -O mldata/' + data + '.mat' + ' ' + url1)
        os.system('wget' + ' -O mldata/' + data + '_gt.mat' + ' ' + url2)
        input_mat = io.loadmat('mldata/' + data + '.mat')[data.lower()]
        target_mat = io.loadmat('mldata/' + data + '_gt.mat')[data.lower() + '_gt']
    target_mat = np.asarray(target_mat, dtype=np.int32)
    input_mat = np.asarray(input_mat, dtype=np.float32)
    return input_mat, target_mat


def Patch(height_index, width_index, MEAN_ARRAY, input_mat):
    """
    Returns a mean-normalized patch, the top left corner of which
    is at (height_index, width_index)

    Inputs:
    height_index - row index of the top left corner of the image patch
    width_index - column index of the top left corner of the image patch

    Outputs:
    mean_normalized_patch - mean normalized patch of size (PATCH_SIZE, PATCH_SIZE)
    whose top left corner is at (height_index, width_index)
    """
    #     transpose_array = np.transpose(input_mat,(2,0,1))
    transpose_array = input_mat
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patch = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patch.shape[0]):
        mean_normalized_patch.append(patch[i] - MEAN_ARRAY[i])

    return np.array(mean_normalized_patch)

def preprocess_data(input_mat, target_mat):
    # print("input_mat dimensions",input_mat.shape)
    HEIGHT = input_mat.shape[0]
    WIDTH = input_mat.shape[1]
    BAND = input_mat.shape[2]
    input_mat = input_mat.astype(np.float64)
    input_mat -= np.min(input_mat)
    input_mat /= np.max(input_mat)

    MEAN_ARRAY = np.ndarray(shape=(BAND,), dtype=np.float64)
    new_input_mat = []
    input_mat = np.transpose(input_mat, (2, 0, 1))
    # print(input_mat.shape)
    for i in range(BAND):
        MEAN_ARRAY[i] = np.mean(input_mat[i, :, :])
        try:
            new_input_mat.append(np.pad(input_mat[i, :, :], PATCH_SIZE // 2, 'constant', constant_values=0))
        except Exception as e:
            print(str(e))
            new_input_mat = input_mat

    # print(np.array(new_input_mat).shape)

    input_mat = np.array(new_input_mat)

    preprocessed_img = np.zeros((HEIGHT, WIDTH, BAND, PATCH_SIZE, PATCH_SIZE), dtype=np.float64)

    for i in range(HEIGHT):
        for j in range(WIDTH):
            curr_inp = Patch(i, j, MEAN_ARRAY, input_mat)
            preprocessed_img[i, j] = curr_inp

    return preprocessed_img


def run_preprocessing():
    input_mat, target_mat = load_data()
    print("preprocessing data ....")
    preprocessed_img = preprocess_data(input_mat, target_mat)
    preprocessed_data = {}
    preprocessed_img = np.asarray(preprocessed_img, dtype=np.float32)
    target_mat = np.asarray(target_mat, dtype=np.int32)
    print("saving data ....")
    preprocessed_data["preprocessed_img"] = preprocessed_img
    preprocessed_data["preprocessed_gt"] = target_mat
    scipy.io.savemat("mldata/" + data + "_Preprocessed_patch_" + str(PATCH_SIZE) + ".mat", preprocessed_data)

if __name__ == '__main__':
    run_preprocessing()
