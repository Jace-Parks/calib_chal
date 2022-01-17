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

#97.03
#use zero as verification 
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


flength = 50
totalframes = 4000 
totalpvacc = 0
totalyvacc = 0 
subalpvacc = 0 
subalyvacc = 0 
traincount = 0
seed(datetime.now())

frames = np.empty((flength,763002),dtype=np.float32)
frame = np.empty((763002),dtype=np.float32)

labels = np.empty((flength,2),dtype=np.float32)
label = np.empty((1,2),dtype=np.float32)

prelabels = np.empty((1198,2),dtype=np.float32)
testframes = np.empty((1198,763002),dtype=np.float32)

verdata = np.empty((100,763002),dtype=np.float32)
verlabs = np.empty((int(totalframes/flength),2),dtype=np.float32)

#create array for saving vacc
#   - pitch
#   - yaw

#create one for loss



index = 0

cap0 = cv2.VideoCapture("labeled/0.hevc")

ret1, cframe = cap0.read()


lname  = "labeled/0.txt"
with open(lname) as f:
    dataset0 = loadtxt(lname, delimiter=' ')


while index < 100:
    sframes = 0
    print("video: ","0"," frame: ",index)
    image = cv2.resize(cframe, (582,437))
    cnewframe = image.reshape((763002))
    frame = cnewframe.astype('float32')
    frame /= 255    

    verdata[index] = frame 

                
    ret1, cframe = cap0.read()
    index += 1



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

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


#training with boot strapping
i = 0
trainedframes = 0
index = 0 
trainindex = 0

while i < totalframes:
    trainindex = 0
    while trainedframes < 50:
        vidnum = randint(1,4)
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

        #cv2.imshow('image window', frame)
        #add wait key. window waits until user presses a key
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #augmentations
        #brightness

        brightval = randint(0,120)
        frame = increase_brightness(frame,value=brightval)

        #cv2.imshow('image window', frame)
        #add wait key. window waits until user presses a key
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

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
    answer = model.predict(verdata)

    print(answer)

    #calculate average error and place that val in an array
    #want to plot loss, and then accuracy of pitch and yaw after each iteration

    #100 - (|measured - given| / given) = accuracy of the answer 

    print(len(answer))
    print(len(dataset0))
    for x in range(100):
        subalpvacc += (100 - abs((abs(answer[x][0] - dataset0[x][0]) / dataset0[x][0]))) 
        subalyvacc += (100 - abs((abs(answer[x][1] - dataset0[x][1]) / dataset0[x][1])))

    pitch = np.float32(subalpvacc / 100)
    yaw = np.float32(subalyvacc / 100)

    print("before",pitch,yaw)
    print(verlabs)

    if pitch <= 0:
        verlabs[traincount][0] = 0.0
    else:
        verlabs[traincount][0] = pitch

    if yaw <= 0:
        verlabs[traincount][1] = 0.0
    else:
        verlabs[traincount][1] = yaw
        
    subalpvacc = 0
    subalyvacc = 0 
    traincount += 1

    print(subalpvacc / 1200)
    print(subalyvacc / 1200)

    print(verlabs)

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

print("success!")

