import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging
import cv2
import math
import keras
import gc
import time
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

def cb(image, beta):
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.0

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

    return new_image

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def getmod():
    model = Sequential()

    #150 beforeho
    model.add(Dense(210, input_dim=763002, activation='LeakyReLU',kernel_initializer='random_normal',bias_initializer='random_normal'))

    model.add(Dense(180, activation='LeakyReLU',kernel_initializer='random_normal',bias_initializer='random_normal'))
    model.add(Dense(150, activation='LeakyReLU',kernel_initializer='random_normal',bias_initializer='random_normal'))
    model.add(Dense(130, activation='LeakyReLU',kernel_initializer='random_normal',bias_initializer='random_normal'))
    model.add(Dense(100, activation='LeakyReLU',kernel_initializer='random_normal',bias_initializer='random_normal'))
    model.add(Dense(80, activation='LeakyReLU',kernel_initializer='random_normal',bias_initializer='random_normal'))
    model.add(Dense(50, activation='LeakyReLU',kernel_initializer='random_normal',bias_initializer='random_normal'))
    model.add(Dense(20, activation='LeakyReLU',kernel_initializer='random_normal',bias_initializer='random_normal'))

    model.add(Dense(2, activation='linear',kernel_initializer='random_normal',bias_initializer='random_normal'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model

def train(model,totalframes,flength,modelnum):
    i = 0
    trainedframes = 0
    index = 0 
    trainindex = 0

    while i < totalframes:
        trainindex = 0
        while trainedframes < flength:
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

            """
            cv2.imshow('image window', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            


            beta = randint(0,100)

            if beta != 0:
                frame = increase_brightness(frame,beta)
            
            
            cv2.imshow('image window', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

            index = frameindex

            if math.isnan(dataset0[index][0]) == False and math.isnan(dataset0[index][1]) == False:
                print("model: ",str(modelnum)," video: ",vidnum," frame: ",frameindex, "i: ",i)
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
    return model
    
def getresults(model,num):
    cap0 = cv2.VideoCapture("unlabeled/" + str(num) + ".hevc")

    ret1, cframe = cap0.read()

    index = 0

    while index < 1200:
        sframes = 0
        print("video: ",str(num)," frame: ",index)
        image = cv2.resize(cframe, (582,437))
        cnewframe = image.reshape((763002))
        frame = cnewframe.astype('float32')
        frame /= 255    

                
        ret1, cframe = cap0.read()
        
        testframe[0] = frame 

        answer = model.predict(testframe)

        print(answer)

        prelabels[index][0] = answer[0][0] 
        prelabels[index][1] = answer[0][1]
        index += 1
            
            
    
    print(prelabels)

    return prelabels



#label = np.empty((1,2),dtype=np.float32)


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

flength = 25
totalframes = 5000
number_models = 4
seed(datetime.now())

start = time.time()

frames = np.empty((flength,763002),dtype=np.float32)
frame = np.empty((763002),dtype=np.float32)
labels = np.empty((flength,2),dtype=np.float32)

model1 = getmod()
model1 = train(model1,totalframes,flength,1)

del frames
del frame
del labels
gc.collect()

prelabels = np.empty((1200,2),dtype=np.float32)
testframe = np.empty((1,763002),dtype=np.float32)

answers51 = getresults(model1,5)
answers61 = getresults(model1,6)
answers71 = getresults(model1,7)
answers81 = getresults(model1,8)
answers91 = getresults(model1,9)

del prelabels
#del testframes
del model1
gc.collect()

frames = np.empty((flength,763002),dtype=np.float32)
frame = np.empty((763002),dtype=np.float32)
labels = np.empty((flength,2),dtype=np.float32)

model2 = getmod()
model2 = train(model2,totalframes,flength,2)

del frames
del frame
del labels
gc.collect()

prelabels = np.empty((1200,2),dtype=np.float32)
#testframes = np.empty((1200,763002),dtype=np.float32)

answers52 = getresults(model2,5)
answers62 = getresults(model2,6)
answers72 = getresults(model2,7)
answers82 = getresults(model2,8)
answers92 = getresults(model2,9)

del prelabels
#del testframes
del model2
gc.collect()

frames = np.empty((flength,763002),dtype=np.float32)
frame = np.empty((763002),dtype=np.float32)
labels = np.empty((flength,2),dtype=np.float32)

model3 = getmod()
model3 = train(model3,totalframes,flength,3)

del frames
del frame
del labels
gc.collect()

prelabels = np.empty((1200,2),dtype=np.float32)

answers53 = getresults(model3,5)
answers63 = getresults(model3,6)
answers73 = getresults(model3,7)
answers83 = getresults(model3,8)
answers93 = getresults(model3,9)

del prelabels
#del testframes
del model3
gc.collect()

frames = np.empty((flength,763002),dtype=np.float32)
frame = np.empty((763002),dtype=np.float32)
labels = np.empty((flength,2),dtype=np.float32)

model4 = getmod()
model4 = train(model4,totalframes,flength,4)

del frames
del frame
del labels
gc.collect()

prelabels = np.empty((1200,2),dtype=np.float32)
testframe = np.empty((1,763002),dtype=np.float32)

answers54 = getresults(model4,5)
answers64 = getresults(model4,6)
answers74 = getresults(model4,7)
answers84 = getresults(model4,8)
answers94 = getresults(model4,9)

del prelabels
#del testframes
del model4
gc.collect()

prelabels = np.empty((1200,2),dtype=np.float32)

for x in range(1200):
    prelabels[x][0] = ((answers51[x][0] + answers52[x][0] + answers53[x][0] +  answers54[x][0]) / 4)
    prelabels[x][1] = ((answers51[x][1] + answers52[x][1] + answers53[x][1] +  answers54[x][1]) / 4)

np.savetxt("unlabeled/5.txt", prelabels)

for x in range(1200):
    prelabels[x][0] = ((answers61[x][0] + answers62[x][0] + answers63[x][0] + answers64[x][0]) / 4)
    prelabels[x][1] = ((answers61[x][1] + answers62[x][1] + answers63[x][1] + answers64[x][1]) / 4)

np.savetxt("unlabeled/6.txt", prelabels)

for x in range(1200):
    prelabels[x][0] = ((answers71[x][0] + answers72[x][0] + answers73[x][0] + answers74[x][0]) / 4)
    prelabels[x][1] = ((answers71[x][1] + answers72[x][1] + answers73[x][1] + answers74[x][1]) / 4)

np.savetxt("unlabeled/7.txt", prelabels)

for x in range(1200):
    prelabels[x][0] = ((answers81[x][0] + answers82[x][0] + answers83[x][0] + answers84[x][0]) / 4)
    prelabels[x][1] = ((answers81[x][1] + answers82[x][1] + answers83[x][1] + answers84[x][1]) / 4)

np.savetxt("unlabeled/8.txt", prelabels)

for x in range(1200):
    prelabels[x][0] = ((answers91[x][0] + answers92[x][0] + answers93[x][0] + answers94[x][0]) / 4)
    prelabels[x][1] = ((answers91[x][1] + answers92[x][1] + answers93[x][1] + answers94[x][1]) / 4)

np.savetxt("unlabeled/9.txt", prelabels)

finished = time.time()
#labeling
#problem with grading script 4.txt last 4 dont exist

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
print("time elasped: ",(finished - start)/60, " minutes")
