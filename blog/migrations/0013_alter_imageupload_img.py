# Generated by Django 3.2.6 on 2021-08-09 23:52

from django.db import migrations
import imagekit.models.fields


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0012_alter_imageupload_img'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageupload',
            name='img',
            field=imagekit.models.fields.ProcessedImageField(upload_to='blog/images.'),
        ),
    ]