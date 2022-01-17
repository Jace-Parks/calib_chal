import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging
import cv2
import math
import keras
import gc
import tensorflow as tf
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import Dense, LSTM
from keras.layers import LeakyReLU
from mpl_toolkits import mplot3d
from random import seed
from random import random
from datetime import datetime
from random import randint
 

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def loadlabels(lname):
    with open(lname) as f:
        dataset0 = loadtxt(lname, delimiter=' ')
    return dataset0


def loadvideo(vname):
    index = 0
    testframes = np.empty((1198,763002),dtype=np.float32)

    cap0 = cv2.VideoCapture(vname)
    ret1, cframe = cap0.read()

    while index < 1198:
        print("video: ",vname[8]," frame: ",index)
        image = cv2.resize(cframe, (582,437))
        cnewframe = image.reshape((763002))
        frame = cnewframe.astype('float32')
        frame /= 255    

        testframes[index] = frame 
       
        ret1, cframe = cap0.read()
        index += 1

    return testframes

def getframe(vname,frameindex):
    cap0 = cv2.VideoCapture(vname)

    cap0.set(cv2.CAP_PROP_POS_FRAMES,frameindex)
    ret1, frame = cap0.read()

def getmodel():
    model = Sequential()
    model.add(Dense(150, input_dim=763002, activation='LeakyReLU'))

    model.add(Dense(110, activation='LeakyReLU'))
    model.add(Dense(100, activation='LeakyReLU'))
    model.add(Dense(80, activation='LeakyReLU'))
    model.add(Dense(50, activation='LeakyReLU'))
    model.add(Dense(30, activation='LeakyReLU'))

    model.add(Dense(2, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model


def writelabels(lname,answer):
    np.savetxt(lname,answer)


testframes = np.empty((1198,763002),dtype=np.float32)
flength = 50
totalframes = 4000 
seed(datetime.now())




