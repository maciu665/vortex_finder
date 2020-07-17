import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import requests
import urllib
import os
import sys
from PIL import Image
import cv2
import math

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
#import keras
#import keras.preprocessing.image
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2500)])

import json
#print(os.getcwd())

os.chdir("/home/maciej/DANE/UDEMY/DL01/retina_faces")

j = json.loads(open('wiry.json').readline())
print(j)
df = pd.read_json('wiry.json', lines=True)
#df.head()
print(df.shape)

from glob import glob
model_paths = glob('snapshots/resnet50_csv_*.h5')
latest_path = sorted(model_paths)[-1]
print("path:", latest_path)
#sys.exit()

model = models.load_model(latest_path, backbone_name='resnet50')
model = models.convert_model(model)

imfiles = []
for i in range(0,1000):
    imfiles.append("/home/maciej/DANE/CFD/DOSTAWCZAK/LICA/lica1.%s.png"%(str(i).zfill(4)))
    #imfiles.append("/home/maciej/DANE/CFD/272/ALIC.%s.png"%i)

#imfiles = ["/home/maciej/DANE/CFD/272/ALIC.1000.png","/home/maciej/DANE/CFD/272/ALIC.1001.png"]
results = {}
results['series'] = {}

ima = []

for imfile in imfiles:
    im = np.array(Image.open(imfile))
    im = im[:,:,:3]
    imp = preprocess_image(im)
    imp, scale = resize_image(im)
    ima.append(imp)
    results['resolution'] = im.shape
    results['scaled'] = imp.shape
    results['scale'] = scale

ima = np.array(ima)

threshold = 0.3
halfsize = 32

for ni in range(int(len(ima)/10)):
    ibatch = ima[ni*10:ni*10+10]
    bimfiles = imfiles[ni*10:ni*10+10]
    print(ibatch.shape)
    #sys.exit()
    boxes, scores, labels = model.predict_on_batch(ibatch)
    # standardize box coordinates
    boxes /= scale
    print("BOXES",len(boxes))

    for i in range(len(ibatch)):
        imfile = bimfiles[i]
        results["series"][imfile] = []
        boxset = boxes[i]
        scoreset = scores[i]
        for b,score in zip(boxset,scoreset):
            if score < threshold:
              break
            if (float(b[0])-(2*halfsize)) > 0:
                mx = float(b[0])+halfsize
            else:
                mx = float(b[2])-halfsize
            if (float(b[1])-(2*halfsize)) > 0:
                my = float(b[1])+halfsize
            else:
                my = float(b[3])-halfsize
            #results["series"][imfile].append({'x0': float(b[0]),'y0':float(b[1]),'x1': float(b[2]),'y1':float(b[3]),'mx':mx,'my':my})
            results["series"][imfile].append({'mx':round(mx,1),'my':round(my,1)})

with open('lica1.json', 'w') as outfile:
    json.dump(results, outfile)
