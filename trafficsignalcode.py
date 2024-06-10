# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:57:30 2024

@author: Rajesh
"""

# loading general packages and libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os
import cv2
import random
from PIL import Image

# loading tensor flow libraries needed
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Dropout, Flatten, MaxPool2D,BatchNormalization
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam,SGD
import keras
from keras.preprocessing import image

seed=1
np.random.seed(seed)
tf.random.set_seed(seed)

train_dir =r"C:\Users\Rajesh\Downloads\trsfficsign\traffic_Data\DATA//" 

# reading the csv file
labels = pd.read_csv(r"C:\Users\Rajesh\Downloads\trsfficsign\labels.csv")
labels

lst = []
for i in labels.index:
    lst.append(len(os.listdir(train_dir + str(i))))
labels['count'] = lst
labels['count'].describe()

# only keep those with enough images in each label 
labels = labels[labels['count'] >= 107.5]
labels

# finding the unknown image
fnames = os.listdir(train_dir + '56')
img = cv2.imread(train_dir + '56/' + fnames[3])
plt.imshow(img)

# renaming
labels.loc[56, "Name"] = "Yield"

labels["Name"] = ["Speed Limit 5", "Speed Limit 40",
       "Speed Limit 60", "Speed Limit 80", "No Left",
       "No Overtake from Left", "No Cars", "No Horn", "Keep Right",
       "Watch for Cars", "Bicycle Crossing", "Zebra Crossing",
       "No Stopping", "No Entry", "Yield"]

# set the image size 
image_size = 128



# input and data augmentation
train_datagen = ImageDataGenerator(
        rescale = 1 / 255.,
        rotation_range=10,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,                                       
        fill_mode="nearest",
        validation_split=0.25,
    )

train_generator = train_datagen.flow_from_directory(
    directory = train_dir,          
    target_size = (image_size, image_size), 
    batch_size = 28,
    shuffle=True,
    class_mode = "categorical",   
    subset = "training"     
)

validation_generator = train_datagen.flow_from_directory(
    directory = train_dir,   
    target_size = (image_size, image_size),   
    batch_size = 28, 
    class_mode = "categorical",
    subset = "validation"
)



from keras.applications.resnet50 import ResNet50


# set classes to number of categories and input weight paths
num_classes = 58
resnet_weights_path = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

# defining the model
model = Sequential()
model.trainable = True


model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(BatchNormalization())
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
        train_generator,
        steps_per_epoch= 60,
        epochs = 5,
        validation_data=validation_generator,
        validation_steps=5)




label = {0:"Speed Limit 5", 1:"Speed Limit 15", 2:"Speed Limit 30", 
         3:"Speed Limit 40", 4:"Speed Limit 50", 5:"Speed Limit 60", 
         6:"Speed Limit 70", 7:"Speed Limit 80", 8:"Don't go straight or left", 
         9:"Don't go straight or right", 10:"Don't go straight", 11:"No Left",
         12:"Don't go right or left", 13:"Don't go right", 14:"No Overtake from Left", 
         15:"No U-turn", 16:"No Cars", 17:"No Horn", 18:"Speed Limit (40km/h)",
         19:"Speed Limit (50km/h)", 20:"Go straight or right", 21:"Watch out for cars",
         22:"Go left", 23:"Go left or right", 24:"Go right", 25:"Keep Left",
         26:"Keep Right", 27:"Roundabout mandatory", 28:"Go Straight",
         29:"Horn", 30:"Bicycle Crossing", 31:"U-turn", 32:"Road Divider",
         33:"Traffic Signals", 34:"Danger ahead", 35:"Zebra Crossing",
         36:"Bicycle Crossing", 37:"Children Crossing", 38:"Dangerous curve to the left",
         39:"Dangerous curve to the right", 40:"Unknown 1", 41:"Unknown 2", 42:"Unknown 3",
         43:"Go right or straight", 44:"Go left or straight", 45:"Unknown 4", 
         46:"Zigzag curve", 47:"Train Crossing", 48:"Under construction", 49:"Unknown 5",
         50:"Fences", 51:"Heavy Vehicle Accidents", 52:"Unknown 6", 53:"Give way",
         54:"No Stopping", 55:"No Entry", 56:"Yield", 57:"Unknown 8"}

img_directory = "/kaggle/input/traffic-sign-dataset-classification/traffic_Data/TEST/021_1_0008.png"
test_image = image.load_img(img_directory, target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image

result = model.predict(test_image)
    
img = mpimg.imread(img_directory)
imgplot = plt.imshow(img)
plt.show()
    
print(f"Predicted class: {label[np.argmax(result)]}")









