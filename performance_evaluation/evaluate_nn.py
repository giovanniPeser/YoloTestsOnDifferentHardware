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
mydevice = 'cpu' #or cuda
model_type = 's'
is_u = True
size=640
yolovers = "5"
yolov8 = False
if(yolovers=="8"):
    yolov8 = True
if(mydevice=='cpu'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if(is_int8_tflite):
    if(yolov8):
        temp = '../yolov8'+model_type+'_saved_model_'+str(size)+'/yolov8'+model_type+'_int8.tflite'
    else:
        if(is_u):
            temp = '../yolov5'+model_type+'u_saved_model_'+str(size)+'/yolov5'+model_type+'u_int8.tflite'
        else:
            temp = '../yolov5'+model_type+'-int8_'+str(size)+'.tflite'
else:
    temp='../yolov5'+model_type+'.pt'
    

    
#model=YOLO(temp)
if((yolov8) or (is_u)):
    model=YOLO(temp)
else:
    model = torch.hub.load(os.getcwd(),'custom',source='local',path=temp, force_reload=True,device=mydevice)
    
image_dir = os.listdir('calibration1000_'+str(size)+'/')
prepro_time = 0
elaboration_time = 0
nm_time = 0
counter = 0
for img in image_dir:
    img_path = 'calibration1000_'+str(size)+'/' + img 
    if((yolov8) or (is_u)):
        result=model(img_path, device=mydevice,imgsz=size)
        elaboration_time += result[0].speed["inference"]
        prepro_time += result[0].speed["preprocess"]
        nm_time += result[0].speed["postprocess"]
    else:
        result=model.forward(img_path,size=size,device=mydevice)
        elaboration_time += result.times[1].dt
        prepro_time += result.times[0].dt
        nm_time += result.times[2].dt
    counter +=1

if((yolov8) or (is_u)):
    print(prepro_time/counter/1000)
    print(elaboration_time/counter/1000)
    print(nm_time/counter/1000)
    print((prepro_time+elaboration_time+nm_time)/counter/1000)
else:
    print(prepro_time/counter)
    print(elaboration_time/counter)
    print(nm_time/counter)
    print((prepro_time+elaboration_time+nm_time)/counter)
print(counter)
print(temp)
