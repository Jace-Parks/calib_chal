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

flength = 50

frames = np.empty((flength,763002),dtype=np.float32)
frame = np.empty((763002),dtype=np.float32)

labels = np.empty((flength,2),dtype=np.float32)
label = np.empty((1,2),dtype=np.float32)

prelabels = np.empty((1200,2),dtype=np.float32)
testframes = np.empty((1200,763002),dtype=np.float32)


#baseline eval
#net 1
#train on all videos 225.06%


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
model.add(Dense(120, input_dim=763002, activation='LeakyReLU'))

model.add(Dense(100, activation='LeakyReLU'))
model.add(Dense(100, activation='LeakyReLU'))
model.add(Dense(80, activation='LeakyReLU'))
model.add(Dense(80, activation='LeakyReLU'))
model.add(Dense(60, activation='LeakyReLU'))

model.add(Dense(2, activation='linear'))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])


#training
for video in range(5):
    vname = "labeled/" + str(video) + ".hevc"
    lname = "labeled/" + str(video) + ".txt"

    cap0 = cv2.VideoCapture(vname)

    with open(lname) as f:
        dataset0 = loadtxt(lname, delimiter=' ')
    
    ret1, frame = cap0.read()

    index = 0
    go = True

    while index < 1200 and go:
        sframes = 0
        go = False
        while sframes < flength and ret1:
            go = True
            print("video: ",video," frame: ",index)
            if math.isnan(dataset0[index][0]) == False and math.isnan(dataset0[index][1]) == False:
                image = cv2.resize(frame, (582,437))
                cnewframe = image.reshape((763002))
                frame = cnewframe.astype('float32')
                frame /= 255    

                frames[sframes] = frame 

                labels[sframes][0] = float(dataset0[index][0])
                labels[sframes][1] = float(dataset0[index][1])
                sframes += 1

            
            ret1, frame = cap0.read()
            index += 1
            if index > 1200:
                break

        model.fit(x = frames, y = labels,epochs=5,batch_size=50)

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