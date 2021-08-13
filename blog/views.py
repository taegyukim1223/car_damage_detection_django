from django.shortcuts import render, redirect
from django.views.generic.detail import DetailView
from .models import Post
from django.views.generic import ListView
from .forms import *
import numpy as np
import pandas as pd
global graph,ans
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from efficientnet_pytorch import EfficientNet
import time

from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
pylab.rcParams['figure.figsize'] = (8.0, 10.0)# Import Libraries

import seaborn as sns
from matplotlib import colors
from tensorboard.backend.event_processing import event_accumulator as ea
from PIL import Image

# Scipy for calculating distance
from scipy.spatial import distance

import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import json, cv2, random
import matplotlib.pyplot as plt
import skimage.io as io

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

# Set base params
plt.rcParams["figure.figsize"] = [16,9]
dataset_dir = 'blog/car_annotation'
val_dir = 'blog/train'
register_coco_instances("car_data", {}, "blog/via_project_19Jul2021_10h37m_coco.json", os.path.join(dataset_dir,val_dir))
# Create your views here.

class PostList(ListView):
    model = Post
    template_name = 'blog/index.html'
    ordering = '-pk'

class PostDetail(DetailView):
    model = Post

# Create your views here.
def upload_image(request):
  
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
  
        if form.is_valid():
            form.save()

    else :
        form = UploadForm()

    return render(request, 'blog/prediction1.html', {'form' : form})
  
  
def image_list(request):
    return render(request, 'blog/list.html', {})

def prediction(request):
    if request.method == 'POST':
        # and request.FILES['myfile']:
        
        # post = request.method == 'POST'
        # myfile = request.FILES['myfile']
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device 객
        car_dir = 'blog/cartegories/'
        categories = pd.read_csv(car_dir+'categories.csv', header = None)
        nb_classes = len(categories)
        categories = categories[0]
        categories = np.array(categories)

        transforms_test = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        data_dir = '_media/'
        test_datasets = datasets.ImageFolder(os.path.join(data_dir,'blog'), transforms_test)
        test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=1, shuffle=True, num_workers=1)

        model = torch.load('blog/Best_model_car_efficient_b4_ver02_25.ph')
        num_features = model.fc.in_features
        # 전이 학습(transfer learning): 모델의 출력 뉴런 수를 6개로 교체하여 마지막 레이어 다시 학습
        model.fc = nn.Linear(num_features, nb_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        model.eval()    
        start_time = time.time()

        # Disable grad
        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0

            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                classes = categories[preds[0]]
                
                result = categories[preds[0]]

        #object detection        
        def damage(view):
            try:
                view[1][list(damage_dict.keys())[0]]
                ans = True 
            except :    
              ans = False
            return ans

        #get configuration
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (damage) + 1
        cfg.MODEL.RETINANET.NUM_CLASSES = 4 # only has one class (damage) + 1
        cfg.MODEL.WEIGHTS = os.path.join("blog/model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
        cfg['MODEL']['DEVICE']='cuda'#or cpu
        damage_predictor = DefaultPredictor(cfg)

        dataset = DatasetCatalog.get("car_data")
        metadata = MetadataCatalog.get("car_data")

        damage_class_map= {0:'dent', 1:'scratch', 2:'destroy'}

        fig, (ax1) = plt.subplots(1, figsize =(16,12))
        im = io.imread('_media/blog/images/123.png')

        #damage inference
        damage_outputs = damage_predictor(im)
        damage_v = Visualizer(im[:, :, ::-1],
                        metadata=metadata, 
                        scale = 1.3,                  
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )

        damage_out = damage_v.draw_instance_predictions(damage_outputs["instances"].to("cpu"))

        damage_prediction_classes = [ damage_class_map[el] + "_" + str(indx) for indx,el in enumerate(damage_outputs["instances"].pred_classes.tolist())]
        damage_polygon_centers = damage_outputs["instances"].pred_boxes.get_centers().tolist()
        damage_dict = dict(zip(damage_prediction_classes,damage_polygon_centers))
        io.imsave('blog/static/detect/detect.png', damage_out.get_image()[:, :, ::-1])
        #plot


        return render(request, "blog/prediction.html", {
                'result': result})

    else:
        return render(request, "blog/prediction.html")