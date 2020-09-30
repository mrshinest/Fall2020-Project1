# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:44:21 2020

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
#from theano import tensor as T
from ntn import *
K.clear_session()
from tensorflow.keras import activations




'''class ntn_layer(keras.layers.Layer):
    def __init__(self, inp_size, out_size, activation='tanh', **kwargs):
          super(ntn_layer, self).__init__(**kwargs)
          self.k = out_size
          self.d = inp_size
          self.activation = activations.get(activation)
          self.test_out = 0

    def build(self,input_shape):
        self.W = self.add_weight(name='w',shape=(self.d, self.d, self.k), initializer='glorot_uniform', trainable=True)
        self.V = self.add_weight(name='v', shape=(self.k, self.d*2), initializer='glorot_uniform', trainable=True)

        self.b = self.add_weight(name='b', shape=(self.k,), initializer='zeros', trainable=True)
        self.U = self.add_weight(name='u', shape=(self.k,), initializer='glorot_uniform',trainable=True)

        super(ntn_layer, self).build(input_shape)

    def call(self ,x ,mask=None):
        e1=x[0] # 实体 1
        e2=x[1] # 实体 2
        batch_size = K.shape(e1)[0]
        V_out, h, mid_pro = [],[],[]
        for i in range(self.k): # 计算内部产品
              V_out = K.dot(self.V[i],K.concatenate([e1,e2]))
              temp = K.dot(e1,self.W[:,:,i])
              h = K.sum(temp*e2,axis=1)
              mid_pro.append(V_out+h+self.b[i])

        tensor_bi_product = K.concatenate(mid_pro,axis=0)
        tensor_bi_product = self.U*self.activation(K.reshape(tensor_bi_product,(self.k,batch_size)))

        self.test_out = K.shape(tensor_bi_product)
        return tensor_bi_product
    def compute_output_shape(self, input_shape):
     return (input_shape[0][0],self.k)'''
#shuffle
def shuffleData(data):
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
xs = np.array(np.reshape(X_train.values,[-1,64,64,1]))
x1s = np.array(X1_train.values)
ys = np.array(y_train.values)

xs1 = X_test.values
x1s1 = X1_test.values
ys1 = y_test.values


#def weight_variable(shape):
#    initial=tf.random.truncated_normal(shape,stddev=0.1)#生成一个截断的正态分布
 #   return tf.Variable(initial)
#def bias_variable(shape):
#    initial=tf.constant(0.1,shape=shape)
#    return tf.Variable(initial)
#??

'''x1=tf.keras.backend.placeholder(shape=(None,3),name='x1')
with tf.name_scope('layer1'):
 
    with tf.name_scope('Wk'):
        Wk = tf.Variable(tf.random.normal(shape=[64,3,3],mean=0.0,stddev=0.001,dtype=tf.float32),name='Wk')
        DT_Wk = tf.transpose(tf.einsum("mj,ijk",x1,Wk),perm=[0, 2, 1])
        DT_Wk_D = tf.linalg.diag_part(tf.einsum("ijk,km", DT_Wk,tf.transpose(x1)))

    with tf.name_scope('Vk'):
        Vk = tf.Variable(tf.random.normal(shape=[64,3],mean=0.0,stddev=0.001,dtype=tf.float32),name='Vk')
        Vk_D = tf.matmul(Vk,tf.transpose(x1))

    with tf.name_scope('bias'):
        b_xfc1 = tf.Variable(tf.random.normal(shape=[64,1],mean=0.0,stddev=0.001,dtype=tf.float32)+0.1,name='b1')

    with tf.name_scope('wx_plus_b1'):
        h_xfc1 = tf.transpose( DT_Wk_D + Vk_D + b_xfc1)

# spatial tiling into 8x8x32
h_xfc1_tile = tf.expand_dims(h_xfc1,1)
h_xfc1_tile = tf.expand_dims(h_xfc1_tile,1)
h_xfc1_tile = tf.tile(h_xfc1_tile,multiples=[1,8,8,1])'''



#1
inpt=Input(shape=(64,64,1))
model1=keras.layers.Conv2D(filters=16, kernel_size=3,strides=(1,1),
                             padding='same',
                             activation='relu',
                             bias_constraint=max_norm(0.10))(inpt)
          
model1=keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='SAME')(model1)   
#2
model1=keras.layers.Conv2D(filters=32, kernel_size=3,strides=(1,1),
                             padding='same',
                             activation='relu',
                             bias_constraint=max_norm(0.10))(model1)
          
model1=keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='SAME')(model1)  
#3
model1=keras.layers.Conv2D(filters=64, kernel_size=3,strides=(1,1),
                             padding='same',
                             activation='relu',
                             bias_constraint=max_norm(0.10))(model1) 
          
model1=keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='SAME')(model1)
input1 = Input(shape=(3,), dtype='float32')
input2 = Input(shape=(3,), dtype='float32')
btp = NeuralTensorLayer(output_dim=64, input_dim=3)([input1, input2])#tile layer output
p = Dense(64)(btp)
p=Lambda(tf.expand_dims, arguments={'axis':(1)})(p)
p=Lambda(tf.expand_dims, arguments={'axis':(1)})(p)
model3=Lambda(tf.keras.backend.tile, arguments={'n':(1, 8, 8, 1)})(p)
model2=keras.layers.concatenate([model1,model3])#merge layer
#4
model2=keras.layers.Conv2D(filters=64, kernel_size=3,strides=(1,1),
                             padding='same',
                             activation='relu',
                             bias_constraint=max_norm(0.10))(model2) 
model2=keras.layers.Conv2D(filters=64, kernel_size=3,strides=(1,1),
                             padding='same',
                             activation='relu',
                             bias_constraint=max_norm(0.10))(model2) 
model2=keras.layers.Conv2D(filters=64, kernel_size=3,strides=(1,1),
                             padding='same',
                             activation='relu',
                             bias_constraint=max_norm(0.10))(model2) 
model2=keras.layers.Conv2D(filters=64, kernel_size=3,strides=(1,1),
                             padding='same',
                             activation='relu',
                             bias_constraint=max_norm(0.10))(model2)           
 #5 
model2=keras.layers.Conv2D(filters=64, kernel_size=3,strides=(1,1),
                             padding='same',
                             activation='relu',
                             bias_constraint=max_norm(0.10))(model2) 
#6 
model2=keras.layers.Conv2D(filters=128, kernel_size=3,strides=(1,1),
                             padding='same',
                             activation='relu',
                             bias_constraint=max_norm(0.10))(model2)  
model2=keras.layers.Conv2D(filters=256, kernel_size=3,strides=(1,1),
                             padding='same',
                             activation='relu',
                             bias_constraint=max_norm(0.10))(model2)           
#7

model2=keras.layers.Flatten()(model2) 
model2=keras.layers.Dense((256*2*2), activation='relu')(model2) 
model2=keras.layers.Dense((50))(model2)

model_combined=Model([inpt,[input1,input2]],model2)

     
def custom_loss(y_true, y_pred):
    
    return tf.reduce_mean(tf.square(y_true-y_pred))
def custom_opt(y_true,y_pred):
    acc = tf.reduce_mean(abs(y_true-y_pred))
    return acc    
     
opt=tf.keras.optimizers.Adam()        
model_combined.compile(loss=custom_loss,
             optimizer = opt,
             metrics = ["accuracy"])
history1=model_combined.fit([xs,x1s,x1s],ys, batch_size=256, epochs=1000,validation_split=0.1)
model_combined.save("my_model")

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
