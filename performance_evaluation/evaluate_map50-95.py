import torch
import os
import tensorflow as tf
from PIL import Image
from matplotlib.image import imread
import numpy as np
import datetime
from ultralytics import YOLO
from torchvision import transforms


#note that image must be resize for tflie
is_int8_tflite = False
device = 'gpu'
model_type = 'su'
is_u = True
size=640
yolovers = "5"
yolov8 = False
if(yolovers=="8"):
    yolov8 = True
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 16-23 sept
if(is_int8_tflite):
    if(yolov8):
        temp = '../yolov8'+model_type+'_saved_model_'+str(size)+'/yolov8'+model_type+'_int8.tflite'
    else:
        if(is_u):
            temp = '../yolov5'+model_type+'u_saved_model_'+str(size)+'/yolov5'+model_type+'u_int8.tflite'
        else:
            temp = '../yolov5'+model_type+'-int8_'+str(size)+'.tflite'
else:
    temp='../yolov'+yolovers+model_type+'.pt'
    

    
model=YOLO(temp)
#if((yolov8) and (not(is_int8_tflite))):
#    model=YOLO(temp)
#else:
#    model = torch.hub.load(os.getcwd(),'custom',source='local',path=temp, force_reload=True,device='cpu')
metrics = model.val(data="coco128.yaml",imgsz=size)
print("model " + yolovers + model_type + " with image size " + str(size) + " map50-95result :" + str(metrics.box.map))
print("model " + yolovers + model_type + " with image size " + str(size) + " map50 result :" + str(metrics.box.map50))
