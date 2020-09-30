# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 20:57:17 2020

@author: dyx
"""

import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy import interpolate 
from tensorflow import keras
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model    
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

import scipy.stats as stats
K.clear_session()
from tensorflow.keras import activations

class NeuralTensorLayer(Layer):
  def __init__(self, output_dim, input_dim=None, **kwargs):
    self.output_dim = output_dim #k
    self.input_dim = input_dim   #d
    if self.input_dim:
      kwargs['input_shape'] = (self.input_dim,)
    super(NeuralTensorLayer, self).__init__(**kwargs)


  def build(self, input_shape):
    mean = 0.0
    std = 1.0
    # W : k*d*d
    k = self.output_dim
    d = self.input_dim
    initial_W_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(k,d,d))
    initial_V_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(2*d,k))
    self.W = K.variable(initial_W_values)
    self.V = K.variable(initial_V_values)
    self.b = K.zeros((self.input_dim,))
    #self.trainable_weights = [self.W, self.V, self.b]


  def call(self, inputs, mask=None):
    if type(inputs) is not list or len(inputs) <= 1:
      raise Exception('BilinearTensorLayer must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))
    e1 = inputs[0]
    e2 = inputs[1]
    batch_size = K.shape(e1)[0]
    k = self.output_dim
    # print([e1,e2])
    feed_forward_product = K.dot(K.concatenate([e1,e2]), self.V)
    # print(feed_forward_product)
    bilinear_tensor_products = [ K.sum((e2 * K.dot(e1, self.W[0])) + self.b, axis=1) ]
    # print(bilinear_tensor_products)
    for i in range(k)[1:]:
      btp = K.sum((e2 * K.dot(e1, self.W[i])) + self.b, axis=1)
      bilinear_tensor_products.append(btp)
    result = K.tanh(K.reshape(K.concatenate(bilinear_tensor_products, axis=0), (batch_size, k)) + feed_forward_product)
    # print(result)
    return result


  def compute_output_shape(self, input_shape):
    # print (input_shape)
    batch_size = input_shape[0][0]
    return (batch_size, self.output_dim)

   


'''def shuffleData(data):
    data_rand = data.sample(frac=1,axis=0).reset_index(drop=True)
    X = data_rand.iloc[:,0:4096]
    X1 = data_rand.iloc[:,4096:4099]

    
    y = data_rand.iloc[:,4099::]
    return X,X1,y
splitted_data = scio.loadmat(r'C:\Users\dyx\Downloads\New folder\train_test_split.mat')
data_train = splitted_data.get('data_train')
data_test = splitted_data.get('data_test')
data_train = pd.DataFrame(data=data_train[0:,0:],    # values
                          index=data_train[0:,0],    # 1st column as index
                          columns=data_train[0,0:])  # 1st row as the column names
data_test = pd.DataFrame(data=data_test[0:,0:],    # values
                          index=data_test[0:,0],    # 1st column as index
                          columns=data_test[0,0:])  # 1st row as the column names


X_train,X1_train,y_train = shuffleData(data_train)
X_test,X1_test,y_test = shuffleData(data_test)
# dataframe to matrix
#xs = tf.reshape(X_train.values,[-1,64,64,1])
x1s = X1_train.values
ys = y_train.values

input1 = Input(shape=(3,), dtype='float32')
input2 = Input(shape=(3,), dtype='float32')
btp = NeuralTensorLayer(output_dim=64, input_dim=3)([input1, input2])
p = Dense(64)(btp)
print(p)
#model = tf.keras.Model([input1, input2],[p])
#sgd = SGD(lr=0.0000000001, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer="adam")
#model.fit([x1s, x1s], ys, nb_epoch=50, batch_size=5)


model = keras.models.Sequential()
model.add(Dense(32, input_dim=32))

model.add(Lambda(tf.expand_dims, arguments={'axis':(1)}))
model.add(Lambda(tf.expand_dims, arguments={'axis':(1)}))
model.add(Lambda(tf.keras.backend.tile, arguments={'n':(1, 64, 64, 1)}))
model.output_shape'''
