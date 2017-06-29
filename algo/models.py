from django.db import models
import datetime

# Create your models here.


class Results(models.Model):
    task_id = models.CharField(max_length=100, blank=True, null=True)
    result_file_name = models.TextField(blank=True, null=True)
    error = models.TextField(blank=True,null=True)
    percentage_done = models.CharField(max_length=20,default='0',blank=False,null=False)
    status_text = models.CharField(max_length=30,default='processing',blank=False,null=False)
    is_done = models.BooleanField(default=False)
    
    class Meta:
        db_table = 'results'

