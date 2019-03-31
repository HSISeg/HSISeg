from django.db import models
import datetime

# Create your models here.


class Results(models.Model):
    pid = models.CharField(max_length=100, blank=True, null=True)
    result_file_name = models.TextField(blank=True, null=True)
    error = models.TextField(blank=True,null=True)
    percentage_done = models.CharField(max_length=20,default='0',blank=False,null=False)
    status_text = models.CharField(max_length=30,default='processing',blank=False,null=False)
    is_done = models.BooleanField(default=False)
    
    class Meta:
        db_table = 'results'

class PUstats(models.Model):
	pos_class = models.CharField(max_length=70, blank=True, null=True, db_index=True)
	neg_class = models.CharField(max_length=70, blank=True, null=True)
	precision = models.FloatField(default=0)
	recall = models.FloatField(default=0)
	temperature = models.FloatField(blank=True, null=True)
	baseline = models.FloatField(blank=True, null=True)
	true_pos = models.IntegerField(blank=True, null=True)
	true_neg = models.IntegerField(blank=True, null=True)
	false_pos = models.IntegerField(blank=True, null=True)
	false_neg = models.IntegerField(blank=True, null=True)
	test_type = models.CharField(max_length=10, blank=True, null=True)
	exclude_class_indx = models.CharField(max_length=50, blank=True, null=True)
	no_train_pos_labelled = models.IntegerField(blank=True, null=True)
	no_train_pos_unlabelled = models.IntegerField(blank=True, null=True)
	no_train_neg_unlabelled = models.IntegerField(blank=True, null=True)
	visual_result_filename = models.CharField(max_length=100, blank=True, null=True)
	creation_time = models.DateTimeField(default=datetime.datetime.now, blank=True)
	train_neg_pos_ratio = models.FloatField(blank=True, null=True)
	unlabelled_prior = models.FloatField(blank=True, null=True)
	threshold = models.FloatField(blank=True, null=True)
	auc = models.FloatField(blank=True, null=True)
	data_name = models.CharField(max_length=70, blank=True, null=True, db_index=True)
	experiment_number = models.CharField(max_length=10, blank=True, null=True)
	preprocessing_time = models.FloatField(blank=True, null=True)
	experiment_time = models.FloatField(blank=True, null=True)
	convergence_plot_path = models.CharField(max_length=200, blank=True, null=True)
	sampling_model = models.CharField(max_length=100, blank=True, null=True)

	class Meta:
		db_table = 'PUstats'


class PNstats(models.Model):
	pos_class = models.CharField(max_length=70, blank=True, null=True, db_index=True)
	neg_class = models.CharField(max_length=70, blank=True, null=True)
	precision = models.FloatField(default=0)
	recall = models.FloatField(default=0)
	temperature = models.FloatField(blank=True, null=True)
	baseline = models.FloatField(blank=True, null=True)
	true_pos = models.IntegerField(blank=True, null=True)
	true_neg = models.IntegerField(blank=True, null=True)
	false_pos = models.IntegerField(blank=True, null=True)
	false_neg = models.IntegerField(blank=True, null=True)
	test_type = models.CharField(max_length=10, blank=True, null=True)
	exclude_class_indx = models.CharField(max_length=50, blank=True, null=True)
	no_train_pos_labelled = models.IntegerField(blank=True, null=True)
	no_train_pos_unlabelled = models.IntegerField(blank = True, null=True)
	no_train_neg_unlabelled = models.IntegerField(blank=True, null=True)
	visual_result_filename = models.CharField(max_length=100, blank=True, null=True)
	creation_time = models.DateTimeField(default=datetime.datetime.now, blank=True)
	train_pos_neg_ratio = models.FloatField(blank=True, null=True)
	threshold = models.FloatField(blank=True, null=True)
	auc = models.FloatField(blank=True, null=True)
	data_name = models.CharField(max_length=70, blank=True, null=True, db_index=True)
	experiment_number = models.CharField(max_length=10, blank=True, null=True)
	preprocessing_time = models.FloatField(blank=True, null=True)
	experiment_time = models.FloatField(blank=True, null=True)
	convergence_plot_path = models.CharField(max_length=200, blank=True, null=True)
	sampling_model = models.CharField(max_length=100, blank=True, null=True)

	class Meta:
		db_table = 'PNstats'

class PatchClass(models.Model):
	class_no = models.IntegerField(blank=True, null=True, db_index=True)
	patch_row_start = models.IntegerField(blank=True, null=True)
	patch_row_end = models.IntegerField(blank=True, null=True)
	patch_col_start = models.IntegerField(blank=True, null=True)
	patch_col_end = models.IntegerField(blank=True, null=True)
	creation_time = models.DateTimeField(default=datetime.datetime.now, blank=True)
	data_name = models.CharField(max_length=70, blank=True, null=True, db_index=True)

	class Meta:
		db_table = 'PatchClass'


