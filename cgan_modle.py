# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 22:05:38 2020

@author: dyx
"""
import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from untitled4 import *
import tensorflow as tf

import os
import scipy.io as scio


combine_data = scio.loadmat(r'C:\Users\dyx\Downloads\New folder\result_combine_H20_S10')
# freq gap 0.6THz
real = combine_data.get('real')[:,670]
imag = combine_data.get('imag')[:,670]
# amp = np.sqrt(np.square(real)+np.square(imag))
# phase = np.arctan2(imag,real)
pattern = combine_data.get('pattern')



# X_train: (60000,28,28,1)
# X_label: (60000,10)
x_train = np.transpose(pattern.reshape(1,28,28,-1))
#on real first
x_label = real.reshape(-1,1)

# mnist: ((64, 64, 1), (10,))
#mnist = tf.data.Dataset.from_tensor_slices((X_train, X_label))
#mnist = mnist.map(lambda x, y: (tf.image.resize_images(x, [64, 64]), y))
# mean centering so (-1, 1)
#mnist = mnist.map(lambda x, y: ((x - 0.5) / 0.5, tf.cast(y, tf.float32)))

# mnist_batch: ((?, 64, 64, 1), (?, 10))
#mnist_batch = mnist.shuffle(1000).batch(BATCH_SIZE)
# feed = mnist_batch_iter.get_next()
# Tensor0: (?, 64, 64, 1)
# Tensor1: (?, 10



latent_dim = 100
n_epoches=10
n_batch=128
#n_classes=128
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
x_train1=x_train
x_label1=x_label
# train model
train(g_model, d_model, gan_model, x_train1,x_label1, latent_dim,10,128)
