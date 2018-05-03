# -*- coding: utf-8 -*-


from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('algo', '0002_auto_20170629_1511'),
    ]

    operations = [
        migrations.AddField(
            model_name='results',
            name='status_text',
            field=models.CharField(default=b'processing', max_length=30),
        ),
    ]
