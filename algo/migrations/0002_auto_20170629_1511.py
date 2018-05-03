# -*- coding: utf-8 -*-


from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('algo', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='results',
            name='error',
            field=models.TextField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name='results',
            name='is_done',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='results',
            name='percentage_done',
            field=models.CharField(default=b'0', max_length=20),
        ),
    ]
