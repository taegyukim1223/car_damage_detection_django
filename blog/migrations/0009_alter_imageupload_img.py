# Generated by Django 3.2.6 on 2021-08-09 17:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0008_auto_20210809_1654'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageupload',
            name='img',
            field=models.FileField(blank=True, null=True, upload_to='blog/car_images/'),
        ),
    ]
