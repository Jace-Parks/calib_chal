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


flength = 50
totalframes = 5000
seed(datetime.now())

frames = np.empty((flength,763002),dtype=np.float32)
frame = np.empty((763002),dtype=np.float32)

labels = np.empty((flength,2),dtype=np.float32)
label = np.empty((1,2),dtype=np.float32)

prelabels = np.empty((1200,2),dtype=np.float32)
testframes = np.empty((1200,763002),dtype=np.float32)



#baseline eval
#net 1
#train on all videos batch 75, epoch 5: 225.06%
#bootstrapping  b=5 s=50 frames 6000 147.54%
#bootstrapping  b=5 s=50 frames 7000 ~160%

#neural net v2:
#~75%
#added an extra layer made did descend slower
#increased toplayer and lower layer neuron count
#~54% with epoch of 5, may be overfitting data


#plotting data for diemensionality reduction 
"""
fig = plt.figure()
ax = plt.axes(projection='3d')

zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
"""

model = Sequential()
model.add(Dense(150, input_dim=763002, activation='LeakyReLU'))

model.add(Dense(110, activation='LeakyReLU'))
model.add(Dense(100, activation='LeakyReLU'))
model.add(Dense(80, activation='LeakyReLU'))
model.add(Dense(50, activation='LeakyReLU'))
model.add(Dense(30, activation='LeakyReLU'))

model.add(Dense(2, activation='linear'))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])


#training with boot strapping
i = 0
trainedframes = 0
index = 0 
trainindex = 0

while i < totalframes:
    trainindex = 0
    while trainedframes < 50:
        vidnum = randint(0,4)
        if vidnum == 4:
            frange = 1195
        else:
            frange = 1199

        vname = "labeled/" + str(vidnum) + ".hevc"
        lname = "labeled/" + str(vidnum) + ".txt"

        cap0 = cv2.VideoCapture(vname)

        with open(lname) as f:
            dataset0 = loadtxt(lname, delimiter=' ')

        frameindex = randint(0,frange)

        index = 0
        flag = True

        cap0.set(cv2.CAP_PROP_POS_FRAMES,frameindex)
        ret1, frame = cap0.read()

        index = frameindex

        if math.isnan(dataset0[index][0]) == False and math.isnan(dataset0[index][1]) == False:
            print("video: ",vidnum," frame: ",frameindex, "i: ",i)
            image = cv2.resize(frame, (582,437))
            cnewframe = image.reshape((763002))
            frame = cnewframe.astype('float32')
            frame /= 255    

            frames[trainindex] = frame 

            labels[trainindex][0] = float(dataset0[index][0])
            labels[trainindex][1] = float(dataset0[index][1])

            trainindex += 1
            trainedframes += 1
            i += 1 
        
    model.fit(x=frames,y=labels,epochs=5,batch_size=50)
    trainedframes = 0
    

del frames
gc.collect()
    
#labeling
#problem with grading script 4.txt last 4 dont exist
for x in range(5):
    cap0 = cv2.VideoCapture("unlabeled/" + str(x+5) + ".hevc")

    ret1, cframe = cap0.read()

    index = 0

    while index < 1199:
            sframes = 0
            print("video: ",x+5," frame: ",index)
            image = cv2.resize(cframe, (582,437))
            cnewframe = image.reshape((763002))
            frame = cnewframe.astype('float32')
            frame /= 255    

            testframes[sframes] = frame 

            sframes += 1

                
            ret1, cframe = cap0.read()
            index += 1
            
            
    answer = model.predict(testframes)
    print(answer)

    np.savetxt("unlabeled/" + str(x+5) + ".txt", answer)
    
    del answer
    gc.collect()

    index = 0
    totalacc0 = 0
    totalacc1 = 0
"""
for x in range(len(dataset0)):
    if math.isnan(dataset0[index][0]) == False and math.isnan(dataset0[index][1]) == False:
        totalacc0 += (abs((answer[x][0] - dataset0[x][0]) / dataset0[x][0]))
        totalacc1 += (abs((answer[x][1] - dataset0[x][1]) / dataset0[x][1]))

        #print("accuracy0: ",100 - ((abs(abs(answer[x][0]) - abs(dataset0[x][0])) / abs(dataset0[x][0])) * 100))
        #print(answer[x][0]," ",dataset0[x][0])
        #print("accuracy1: ",100 - ((abs(abs(answer[x][1]) - abs(dataset0[x][1])) / abs(dataset0[x][1])) * 100))
        #print(answer[x][1]," ",dataset0[x][1])
        print("index: ",index)
        print("totals: ",totalacc0, " ",totalacc1)

        index += 1

print("first avg: ",totalacc0)
print("second avg: ",totalacc1)
print("first avg: ",totalacc0 / index)
print("second avg: ",totalacc1 / index)
"""

print("success!")

