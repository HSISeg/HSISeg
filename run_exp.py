import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "HSISeg.settings")
django.setup()
from semiSuper.exp_preprocessing import get_preprocessed_data
from semiSuper.PU_exp import run_human_guess, run_prior_calculate

clust_labelled_img, clust_prob_labelled_img, preprocessed_img, target_mat = get_preprocessed_data()


run_human_guess(clust_labelled_img, preprocessed_img, target_mat)
run_prior_calculate(clust_labelled_img, preprocessed_img, target_mat)