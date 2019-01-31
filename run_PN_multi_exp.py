import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "HSISeg.settings")
django.setup()
from semiSuper.exp_preprocessing import get_preprocessed_data
from semiSuper.PN_multi_step_exp import run_cluster_dist
clust_labelled_img, clust_prob_labelled_img, preprocessed_img, target_mat = get_preprocessed_data()

run_cluster_dist(clust_labelled_img, preprocessed_img, target_mat)