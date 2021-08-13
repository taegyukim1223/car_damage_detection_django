from django.urls import path
from . import views
from .views import *

urlpatterns = [
    path('', views.PostList.as_view()),
    # path('', views.index),
    path('<int:pk>/', views.PostDetail.as_view()),
    path('prediction/', views.upload_image, name = 'upload_image'),
    path('prediction1/', views.prediction, name = 'prediction'),
    path('list/', views.image_list, name = 'image_list'),
]