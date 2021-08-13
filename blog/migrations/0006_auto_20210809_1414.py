# Generated by Django 3.2.6 on 2021-08-09 14:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0005_post_head_image'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='file_upload',
            field=models.FileField(blank=True, upload_to='blog/files/'),
        ),
        migrations.AlterField(
            model_name='post',
            name='head_image',
            field=models.ImageField(blank=True, upload_to='blog/images/'),
        ),
    ]