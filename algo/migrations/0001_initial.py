# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Results',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('task_id', models.CharField(max_length=100, null=True, blank=True)),
                ('result_file_name', models.TextField(null=True, blank=True)),
            ],
            options={
                'db_table': 'results',
            },
        ),
    ]
