#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required packages
from __future__ import print_function, division
from builtins import range, input

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

import os

def mkdir(p):
  if not os.path.exists(p):
    os.mkdir(p)

def link(src, dst):
  if not os.path.exists(dst):
    os.symlink(src, dst, target_is_directory=True)

mkdir('../Data')


classes = ['<50%', '>=50%']

train_path_from = os.path.abspath('Data/Training')
valid_path_from = os.path.abspath('Data/Validation')

train_path_to = os.path.abspath('Data/Training')
valid_path_to = os.path.abspath('Data/Validation')

mkdir(train_path_to)
mkdir(valid_path_to)


for c in classes:
  link(train_path_from + '/' + c, train_path_to + '/' + c)
  link(valid_path_from + '/' + c, valid_path_to + '/' + c)


# In[2]:


image_size = [500, 500]

epochs= 5
batch_size=20

train_path = 'Data/Training'
valid_path = 'Data/Validation'

image_files = glob (train_path+'/*/*.tif')
valid_image_files = glob (valid_path+'/*/*.tif')

folders= glob (train_path + '/*')

plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()


# In[3]:


vgg= VGG16(input_shape=image_size+[3], weights='imagenet', include_top=False)

#to prevent training the existing weights
for layer in vgg.layers:
    layer.trainable =False

#output layer
x=Flatten()(vgg.output)
prediction= Dense(len(folders), activation='softmax')(x)

# the model
model = Model(inputs=vgg.input, outputs=prediction)

model.summary()


# In[4]:


model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

test_gen = gen.flow_from_directory(valid_path, target_size=image_size)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
    labels[v] = k

for x, y in test_gen:
    print("min:", x[0].min(), "max:", x[0].max())
    plt.title(labels[np.argmax(y[0])])
    plt.imshow(x[0])
    plt.show()
    break


# In[ ]:


# create generators for train and valid datasets. 
train_generator = gen.flow_from_directory(train_path, target_size = image_size, shuffle=True, batch_size =batch_size)
valid_generator = gen.flow_from_directory(valid_path, target_size = image_size, shuffle=True, batch_size =batch_size)

# fit the model

r = model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    steps_per_epoch=len(image_files) // batch_size,
    validation_steps=len(valid_image_files) // batch_size,
)


# In[ ]:


def get_confusion_matrix(data_path, N):
    print("Generating confusion matrix", N)
    predictions = []
    targets = []
    i = 0
    for x, y in gen.flow_from_directory(data_path, target_size=image_size, shuffle=False, batch_size=batch_size * 2):
        i += 1
        if i % 50 == 0:
            print(i)
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        if len(targets) >= N:
            break
            
    cm = confusion_matrix(targets, predictions)
    return cm


cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
print(valid_cm)

# plot some data

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()

from utils import plot_confusion_matrix
plot_confusion_matrix(cm, labels, title='Train confusion matrix')
plot_confusion_matrix(valid_cm, labels, title='Validation confusion matrix')

