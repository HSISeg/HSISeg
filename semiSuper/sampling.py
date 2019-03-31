import numpy as np
import Config, math
import queue

# P(cluster|X) , X is the event that a random pixel is positive
def get_clust_given_pos_prob(clust_labelled_img, train_lp_pixels, cross_pos_pixels):
    pos_sel = np.zeros(clust_labelled_img.shape, dtype=np.bool)
    pos_sel[train_lp_pixels] = True
    pos_sel[cross_pos_pixels] = True
    n_pos = len(train_lp_pixels[0]) + len(cross_pos_pixels[0])
    n_classes = np.max(clust_labelled_img) + 1
    clust_pos_prob = np.zeros(n_classes, dtype=np.float32)
    for i in range(n_classes):
        clust_pos_prob[i] = ((np.logical_and(clust_labelled_img == i, pos_sel)).sum() + Config.epsilon) / n_pos
    return clust_pos_prob

# P(cluster)
def get_prob_cluster(clust_labelled_img):
    n_classes = np.max(clust_labelled_img) + 1
    size = clust_labelled_img.size
    clust_prob = np.zeros(n_classes, dtype=np.float32)
    for i in range(n_classes):
        clust_prob[i] = len(np.where(clust_labelled_img == i)[0]) / size
    return clust_prob

def get_euclidean_dist(point1, point2):
    s = math.pow((point1[1] - point2[1]), 2) + math.pow((point1[0] - point2[0]), 2)
    return s ** 0.5

# P(X), X is the event that a random pixel is positive
def get_distance_from_positive(train_lp_pixels, cross_pos_pixels, length, width,  baseline, temp):
    all_pos_pixels = (np.concatenate((train_lp_pixels[0], cross_pos_pixels[0]), axis=0), np.concatenate((train_lp_pixels[1], cross_pos_pixels[1]), axis=0))
    dist = np.zeros((length, width), dtype=np.float32)
    for i in range(length):
        for j in range(width):
            dist[i][j] = min([get_euclidean_dist((i, j), (all_pos_pixels[0][l], all_pos_pixels[1][l])) for l in range(len(all_pos_pixels[0]))])
    dist = 1 / (1 + np.exp((dist - baseline)/temp))
    return dist

# P(X|c), probability of a random pixel being positive given that it belongs to a cluster
def get_point_wise_prob(clust_labelled_img, train_lp_pixels, cross_pos_pixels, is_dist_based, baseline, temp, is_uniform_sampling=False):
    final_prob = np.zeros(clust_labelled_img.shape, dtype=np.float32)
    if not is_uniform_sampling:
        if is_dist_based:
            # P(X)
            dist = get_distance_from_positive(train_lp_pixels, cross_pos_pixels, clust_labelled_img.shape[0], clust_labelled_img.shape[1],  baseline, temp)
        else:
            # P(X)
            dist = np.ones(clust_labelled_img.shape, dtype=np.float32)
        # P(cluster|X)
        clust_pos_prob = get_clust_given_pos_prob(clust_labelled_img, train_lp_pixels, cross_pos_pixels)
        # P(cluster)
        clust_sel_prob = get_prob_cluster(clust_labelled_img)
        n_classes = np.max(clust_labelled_img) + 1
        
        for i in range(n_classes):
            class_pixels = clust_labelled_img == i
            # bayes rule
            final_prob[class_pixels] = (dist[class_pixels] * clust_pos_prob[i])/ clust_sel_prob[i]
    final_prob = final_prob / np.sum(final_prob)
    final_prob = 1 - final_prob
    return final_prob

def get_pos_pixels(pos_class_list, gt_labelled_img, train_pos_percentage, cross_pos_percentage):
    pos_pixels = np.where(np.isin(gt_labelled_img, pos_class_list) == True)
    is_random = Config.is_random_positive_sampling
    if len(pos_pixels[0]) == 0:
        raise ValueError("no positive pixels in the image.")
    if train_pos_percentage + cross_pos_percentage > 100:
        raise ValueError(" pos_percentage of train and pos_percentage of cross validation together can't be greater than 100 ")
    n_pos_pixels = len(pos_pixels[0])
    n_train_pos_pixels = (n_pos_pixels * train_pos_percentage) // 100
    # cross validation
    n_cross_pos_pixels = (n_pos_pixels * cross_pos_percentage) // 100
    # n_train_pos_pixels = 200
    if n_train_pos_pixels == 0:
        raise ValueError("no positive pixels for training.")

    indx = np.random.permutation(len(pos_pixels[0]))
    if is_random:
        train_lp_pixels = (pos_pixels[0][indx[:n_train_pos_pixels]], pos_pixels[1][indx[:n_train_pos_pixels]])
        # cross validation
        cross_pos_pixels = (pos_pixels[0][indx][n_train_pos_pixels: n_train_pos_pixels + n_cross_pos_pixels],
                            pos_pixels[1][indx][n_train_pos_pixels: n_train_pos_pixels + n_cross_pos_pixels])
    else:
        n_positive_selected = 0
        row_indx = []
        col_indx = []
        q = queue.Queue()
        q.put((pos_pixels[0][0], pos_pixels[1][0]))
        col_add = [-1, -1, -1, 0, 0, 1, 1, 1]
        row_add = [-1, 0, 1, -1, 1, -1, 0, 1]
        visited = np.zeros(gt_labelled_img.shape, dtype=np.bool)
        while n_positive_selected < n_train_pos_pixels + n_cross_pos_pixels:
            (row, col) = q.get()
            visited[row][col] = True
            row_indx.append(row)
            col_indx.append(col)
            n_positive_selected += 1
            found = False
            for i in range(0, 8):
                col_new = col + col_add[i]
                row_new = row + row_add[i]
                
                if col_new >=0 and col_new < gt_labelled_img.shape[1] and \
                    row_new >=0 and row_new < gt_labelled_img.shape[0] and \
                    not visited[row_new][col_new] and gt_labelled_img[row_new][col_new] in pos_class_list:
                    q.put((row_new, col_new))
                    found = True
            if not found and n_positive_selected < n_train_pos_pixels + n_cross_pos_pixels:
                exclude_pixels = np.copy(visited)
                exclude_pixels[np.logical_not(np.isin(gt_labelled_img, pos_class_list))] = True
                unvistited_pixels = np.where(exclude_pixels == False)
                q.put((unvistited_pixels[0][0], unvistited_pixels[1][0]))
        train_lp_pixels = (np.array(row_indx[:n_train_pos_pixels]), np.array(col_indx[:n_train_pos_pixels]))
        cross_pos_pixels = (np.array(row_indx[n_train_pos_pixels:]), np.array(col_indx[n_train_pos_pixels:]))
    return train_lp_pixels, cross_pos_pixels

def get_exclude_pixels(pos_class_list, neg_class_list, gt_labelled_img):
    exclude_pixels = np.logical_not(np.isin(gt_labelled_img, list(set(pos_class_list).union(set(neg_class_list)))))
    return exclude_pixels
