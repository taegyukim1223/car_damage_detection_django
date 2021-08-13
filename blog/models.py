from django.db import models
from imagekit.models import ProcessedImageField
from imagekit.processors import ResizeToFill

# Create your models here.

class Post(models.Model):
    title = models.CharField(max_length=30)
    content = models.TextField()
    file_upload = models.FileField(upload_to= 'blog/files/', blank = True)
    head_image = models.ImageField(upload_to = 'blog/images/', blank = True)
    created_at = models.DateField(auto_now_add=True)
    updated_at = models.DateField(auto_now=True)
    # author:

    def __str__(self):
        return f'[{self.pk}]{self.title}'

class ImageUpload(models.Model) : 
    img = ProcessedImageField(upload_to='blog/images',
                                           format='PNG',
                                           options={'quality': 100})

    def __str__(self):
        return f'[{self.pk}]{self.title}'

