# Generated by Django 3.2.6 on 2021-08-09 17:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0009_alter_imageupload_img'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageupload',
            name='img',
            field=models.FileField(blank=True, default=1, upload_to='blog/images/'),
            preserve_default=False,
        ),
    ]
