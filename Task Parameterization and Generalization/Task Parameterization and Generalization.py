#!/usr/bin/env python
# coding: utf-8

# In[13]:

import matplotlib.pyplot as plt
import numpy as np

# Simple data to display in various forms
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

f, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')
# In[12]:


plt.close('all')

# In[14]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import numpy as npntime

import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import math
import random
from keras.models import Sequential,Model
from keras.layers import TimeDistributed,Dense,Activation,Layer,Input,Average,Concatenate,Flatten,Lambda
from keras.optimizers import Adam
import pylab as pl
from IPython import display
import numpy as np
import csv
import data_format

# In[15]:


def csv_to_array(csv_file_name):
    data = []
    flag = True
    with open(csv_file_name, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if(flag):
                flag = False
                continue
            data.append(row)
    
    data = np.array(data)


# In[16]:


d_N = 1
n_max=1 # maximum number of observations that can be used during the training process
a,b,c,d = data_format.data_per_trajectory('/home/colors/Desktop/Cappuccino-Preparing-Robot-Baxter/carry_data/train1', True)
train_joints = np.array(a)
train_n = np.array(b)
train_t = np.array(c)  
train_p = np.array(d)


# In[17]:


train_joints.shape


# In[ ]:





# In[18]:


train_p.shape


# In[19]:


def get_train_sample():
    observation = np.zeros((1,n_max,19))
    observation_flag = np.zeros((1,1,n_max))
    target = np.zeros((1,1,3)) # t, ob_p, w_p  
    gamma = random.randint(0,d_N-1)
    ob_p = train_p[gamma,0]
    w_p = train_p[gamma,1]
    obs_n = random.randint(1,n_max)
    
    perm = np.random.permutation(train_n[gamma])
    
    for i in range(obs_n):
        observation[0,i] = [train_t[gamma,perm[i]],
                            ob_p, # 0,1,2 -> 1/3,2/3,1
                            w_p, # 0,1,2,3 -> 1/4,2/4,3/4,1
                            train_joints[gamma,0,perm[i]],
                            train_joints[gamma,1,perm[i]],
                            train_joints[gamma,2,perm[i]],
                            train_joints[gamma,3,perm[i]],
                            train_joints[gamma,4,perm[i]],
                            train_joints[gamma,5,perm[i]],
                            train_joints[gamma,6,perm[i]],
                            train_joints[gamma,7,perm[i]],
                            train_joints[gamma,8,perm[i]],
                            train_joints[gamma,9,perm[i]],
                            train_joints[gamma,10,perm[i]],
                            train_joints[gamma,11,perm[i]],
                            train_joints[gamma,12,perm[i]],
                            train_joints[gamma,13,perm[i]],
                            train_joints[gamma,14,perm[i]],
                            train_joints[gamma,15,perm[i]]
                           ]
        observation_flag[0,0,i] = 1./obs_n
    target[0,0] = [train_t[gamma,perm[obs_n]], ob_p, w_p]
    return [observation,observation_flag,target], \
            [[[[train_joints[gamma,0,perm[obs_n]],
                train_joints[gamma,1,perm[obs_n]],
                train_joints[gamma,2,perm[obs_n]],
                train_joints[gamma,3,perm[obs_n]],
                train_joints[gamma,4,perm[obs_n]],
                train_joints[gamma,5,perm[obs_n]],
                train_joints[gamma,6,perm[obs_n]],
                train_joints[gamma,7,perm[obs_n]],
                train_joints[gamma,8,perm[obs_n]],
                train_joints[gamma,9,perm[obs_n]],
                train_joints[gamma,10,perm[obs_n]],
                train_joints[gamma,11,perm[obs_n]],
                train_joints[gamma,12,perm[obs_n]],
                train_joints[gamma,13,perm[obs_n]],
                train_joints[gamma,14,perm[obs_n]],
                train_joints[gamma,15,perm[obs_n]],
                ]]]],gamma


# In[20]:


def plt_predictions(gamma=1): 
    ob_p=train_p[gamma,0]
    w_p=train_p[gamma,1]
    prediction = np.zeros((16,train_n[gamma]))
    prediction_std = np.zeros((16,train_n[gamma])) #TODO: 6?
    observation = np.zeros((1,n_max,19))
    observation_flag = np.zeros((1,1,n_max))
    target = np.zeros((1,1,3))
    observation[0,0] = [0,ob_p,w_p,
                        train_joints[gamma,0,0],
                        train_joints[gamma,1,0],
                        train_joints[gamma,2,0],
                        train_joints[gamma,3,0],
                        train_joints[gamma,4,0],
                        train_joints[gamma,5,0],
                        train_joints[gamma,6,0],
                        train_joints[gamma,7,0],
                        train_joints[gamma,8,0],
                        train_joints[gamma,9,0],
                        train_joints[gamma,10,0],
                        train_joints[gamma,11,0],
                        train_joints[gamma,12,0],
                        train_joints[gamma,13,0],
                        train_joints[gamma,14,0],
                        train_joints[gamma,15,0]]
    observation_flag[0,0,0] = 1.
    joint_names = ['Base Joint (rad)','Shoulder Joint (rad)','Elbow Joint (rad)','Wrist1 Joint (rad)','Wrist2 Joint (rad)','Wrist3 Joint (rad)',]        
    for i in range(train_n[gamma]):
        target[0,0] = [train_t[gamma,i],ob_p,w_p]
        p = model.predict([observation,observation_flag,target])[0][0]
        prediction[:,i] = p[:16]
        for j in range(16):
            prediction_std[j,i] = math.log(1+math.exp(p[j]))#TODO: dont understand here. did I do right?
    for joint in range(16):
        fig = plt.figure(figsize=(5,5))
        plt.title(joint_names[joint])
        if joint == 4:
            plt.ylim(-1.35,-1.75)
        for i in range(d_N):
            plt.plot(range(train_n[i]),train_joints[i,joint,:train_n[i]])
        plt.plot(range(train_n[gamma]),prediction[joint,:train_n[gamma]],color='red')
        plt.errorbar(range(train_n[gamma]),prediction[joint,:train_n[gamma]],yerr=prediction_std[2,:train_n[gamma]],color = 'red',alpha=0.1)
        plt.show()


# In[21]:


def custom_loss(y_true, y_pred):
    mean, log_sigma = tf.split(y_pred, 2, axis=-1)
    y_target, temp =tf.split(y_true,2,axis=-1)
    sigma = tf.nn.softplus(log_sigma)
    dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
    loss = -tf.reduce_mean(dist.log_prob(y_target))
    return loss


# In[22]:


observation_layer = Input(shape=(n_max,19))
observation_flag_layer=Input(shape=(1,n_max)) 
observation_encoded = TimeDistributed(Dense(128, activation='relu'))(observation_layer)
observation_encoded = TimeDistributed(Dense(128, activation='relu'))(observation_encoded)
observation_encoded = TimeDistributed(Dense(128, activation='relu'))(observation_encoded)
observation_encoded = TimeDistributed(Dense(128, activation='relu'))(observation_encoded)
observation_encoded = TimeDistributed(Dense(128))(observation_encoded)
matmul_layer=Lambda(lambda x:(tf.matmul(x[0],x[1])), output_shape =(1,128))
representation=matmul_layer([observation_flag_layer,observation_encoded])
target_layer = Input(shape=(1,3))
query_net_input = Concatenate(axis=2)([representation, target_layer])
query = Dense(128, activation='relu')(query_net_input)
query = Dense(128, activation='relu')(query)
query = Dense(128, activation='relu')(query)
query = Dense(128, activation='relu')(query)
output_layer = Dense(16)(query)
model = Model(inputs=[observation_layer,observation_flag_layer,target_layer],outputs=output_layer)
model.compile(optimizer = Adam(lr = 1e-4),loss=custom_loss)
model.summary()


# In[24]:


train_loss = np.zeros(2000)
max_iterations=1000000
for step in range(max_iterations):
    inp,out,gamma = get_train_sample()
    out = np.array(out)
    data = model.fit(inp,out,batch_size=1,verbose=0)

    if step % 1000 == 0:
        train_loss[step//1000] = data.history['loss'][0]
    if step % 10000 == 0:
        display.clear_output(wait=True)
        display.display(pl.gcf())
        print( 'step:', step)
        print( 'loss:', data.history['loss'][0])
        plt.title('Train Loss')
        plt.plot(range(2000),train_loss)
        plt.show()
        mean_loss = np.zeros((100))
        for i in range(100):
            mean_loss[i] = np.mean(train_loss[i*20:(i+1)*20])
        fig = plt.figure()
        plt.title('Train Loss (Smoothed)')
        plt.plot(range(100),mean_loss)
        plt.show()
        plt_predictions(0) #TODO: change gamma here


# In[ ]:





# In[ ]:




