
# coding: utf-8



import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
set_session(tf.Session(config=config))
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D,ZeroPadding2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from sklearn.model_selection import KFold
import os




x=pickle.load(open("7041_setdv.pickle","rb"))





inp=[]
y=[]
for i in range(len(x)):
    inp.append(x[i][0])
    y.append(x[i][1])





# Padding to Make Shape of Input Uniform
final_inp=[]

for v in inp:
    if(v.shape[0]%2==0):
        sp=((111-v.shape[0])/2,((111-v.shape[0])/2+1))
    else:
        sp=((111-v.shape[0])/2,(111-v.shape[0])/2)
    xxx=np.pad(v,pad_width=(sp,(0,0)),mode="constant")
    final_inp.append(xxx)
final_inp=np.array(final_inp).reshape((len(inp), 111, 4096,1))
y=np.array(y)


kfold = KFold(n_splits=10)
vf=1
for train,test in kfold.split(y):
    inputs = Input(shape=(111,4096,1), dtype='float32')

    c0 = Conv2D(200, kernel_size=(3, 4096), padding='valid', kernel_initializer='normal', activation='relu')(inputs)
    c1 = Conv2D(200, kernel_size=(4, 4096), padding='valid', kernel_initializer='normal', activation='relu')(inputs)
    c2 = Conv2D(200, kernel_size=(5, 4096), padding='valid', kernel_initializer='normal', activation='relu')(inputs)

    mx0 = MaxPool2D(pool_size=(111 - 3 + 1, 1), strides=(1,1), padding='valid')(c0)
    mx1 = MaxPool2D(pool_size=(111 - 4 + 1, 1), strides=(1,1), padding='valid')(c1)
    mx2 = MaxPool2D(pool_size=(111- 5+ 1, 1), strides=(1,1), padding='valid')(c2)

    concatenated_tensor = Concatenate(axis=1)([mx0, mx1, mx2])


    flatten = Flatten()(concatenated_tensor)
    dropout1 = Dropout(0.3)(flatten)
    dense=Dense(units=100,activation='relu')(dropout1)
    dense2=Dense(units=50,activation="relu")(dense)
    dropout2 = Dropout(0.5)(dense2)
    output = Dense(units=1, activation='relu')(dropout2)
    model = Model(inputs=inputs, outputs=output)

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])

    checkpoint = ModelCheckpoint('FOLD:'+str(vf)+ ' {epoch:03d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    model.fit(final_inp[train], y[train], batch_size=50, epochs=250, verbose=1,callbacks=[checkpoint], validation_data=(final_inp[test], y[test])) 
    vf+=1