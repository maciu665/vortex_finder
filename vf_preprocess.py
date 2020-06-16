# -*- coding: utf-8 -*-
#!/usr/bin/python
########################################
##  author: maciu665
########################################
## preprocessing images for cnn training
########################################

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

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


import json
#print(os.getcwd())

os.chdir("/home/maciej/DANE/UDEMY/DL01/retina_faces")

j = json.loads(open('wiry.json').readline())
print(j)
df = pd.read_json('wiry.json', lines=True)
#df.head()
print(df.shape)



'''
im = Image.open('/home/maciej/DANE/CFD/272/ALIC.0138.png')
plt.imshow(im)
plt.show()
'''
#sys.exit()

converted_data_train = {
    'image_name': [],
    'x_min': [],
    'y_min': [],
    'x_max': [],
    'y_max': [],
    'class_name': [],
}

converted_data_test = {
    'image_name': [],
    'x_min': [],
    'y_min': [],
    'x_max': [],
    'y_max': [],
    'class_name': [],
}

if not os.path.exists('wiry'):
  os.mkdir('wiry')

idx = 0 # global counter for filenames

def map_to_data(row, converted_data):
  global idx
  #r = requests.get(row['content'])

  tf = open(row['content'],'rb')
  print(row['content'])
  imdata = tf.read()
  tf.close()

  filepath = 'wiry/wir_%s.png' % idx

  # don't bother to overwrite
  if not os.path.exists(filepath):
    with open(filepath, 'wb') as f:
      f.write(imdata)

  # there could be more than 1 face per image
  for anno in row['annotation']:
    converted_data['image_name'].append(filepath)

    width = anno['imageWidth']
    height = anno['imageHeight']
    height = anno['imageWidth']
    width = anno['imageHeight']

    # calculate box coordinates
    x1 = int(round(anno['points'][0]['x'] * height))
    y1 = int(round(anno['points'][0]['y'] * height)) #height
    x2 = int(round(anno['points'][1]['x'] * height))
    y2 = int(round(anno['points'][1]['y'] * height))

    print(y1,y2,width,height)

    if y2 > height:
        #print("koza")
        #print(width, height, y1, y2)
        dy = height - y2
        y2 = y2 + dy
        y1 = y1 - dy
        #print(width, height, y1, y2)
        #sys.exit()



    converted_data['x_min'].append(x1)
    converted_data['y_min'].append(y1)
    converted_data['x_max'].append(x2)
    converted_data['y_max'].append(y2)

    # they are all the same class
    converted_data['class_name'].append('wir')

  # update counter
  idx += 1

#sys.exit()
# we must split BEFORE converting the data
# after converting the data, multiple rows will have the same image
# we won't want to split then3
train_df, test_df = train_test_split(df, test_size=0.2)
print(train_df)
print(test_df)
#sys.exit()

# this will be slow since it has to download all the images

# just in case we run again later
idx = 0

# train
train_df.apply(lambda row: map_to_data(row, converted_data_train), axis=1)

# test
test_df.apply(lambda row: map_to_data(row, converted_data_test), axis=1)

#sys.exit()

# this will overwrite the previous dfs
train_df = pd.DataFrame(converted_data_train)
test_df = pd.DataFrame(converted_data_test)
train_df.head()

train_df.shape

train_df[train_df['image_name'] == 'wiry/wir_1.jpg']

def show_image_with_boxes(df):
  # pick a random image
  filepath = df.sample()['image_name'].values[0]

  # get all rows for this image
  df2 = df[df['image_name'] == filepath]
  im = np.array(Image.open(filepath))

  # if there's a PNG it will have alpha channel
  im = im[:,:,:3]
  img = cv2.UMat(im).get()

  for idx, row in df2.iterrows():
    box = [
      row['x_min'],
      row['y_min'],
      row['x_max'],
      row['y_max'],
    ]
    print(box)
    #img = cv2.UMat(im).get()
    draw_box(img, box, (255, 0, 0))

  plt.axis('off')
  plt.imshow(img)
  plt.show()

show_image_with_boxes(train_df)
#sys.exit()
train_df.to_csv('annotations.csv', index=False, header=None)

classes = ['wir']
with open('classes.csv', 'w') as f:
  for i, class_name in enumerate(classes):
    f.write(f'{class_name},{i}\n')

#!head classes.csv

#!head annotations.csv
sys.exit()
if not os.path.exists('snapshots'):
  os.mkdir('snapshots')

PRETRAINED_MODEL = 'snapshots/_pretrained_model.h5'
'''
URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
urllib.request.urlretrieve(URL_MODEL, PRETRAINED_MODEL)

print('Downloaded pretrained model to ' + PRETRAINED_MODEL)

!keras-retinanet/keras_retinanet/bin/train.py --freeze-backbone \
  --random-transform \
  --weights {PRETRAINED_MODEL} \
  --batch-size 8 \
  --steps 500 \
  --epochs 15 \
  csv annotations.csv classes.csv

'''
