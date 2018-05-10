import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "HSISeg.settings")
django.setup()
from semiSuper.PU_exp import run as run_PU
from semiSuper.exp_preprocessing import get_preprocessed_data

clust_labelled_img, preprocessed_img, target_mat = get_preprocessed_data()

run_PU(clust_labelled_img, preprocessed_img, target_mat)
