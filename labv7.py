import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging
import cv2
import math
import keras
import gc
import time
import sys
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
    
    

    #,kernel_regularizer=keras.regularizers.l2(l2=0.1),bias_regularizer=keras.regularizers.l2(l2=0.1)
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

def train(model,totalframes,flength,modelnum,totalf):
    i = 0
    trainedframes = 0
    index = 0 
    trainindex = 0
    maxtrain  = 1
    framesleft = totalframes

    #setting inital training stuff
    if framesleft >= maxtrain:
        frames = np.empty((maxtrain,763002),dtype=np.float32)
        labels = np.empty((maxtrain,2),dtype=np.float32)
        flength = maxtrain
    else:
        frames = np.empty((framesleft,763002),dtype=np.float32)
        labels = np.empty((framesleft,2),dtype=np.float32)
        flength = framesleft      

    while i < totalframes:
        trainindex = 0

        if i != 0:
            if framesleft >= maxtrain:
                frames = np.empty((maxtrain,763002),dtype=np.float32)
                labels = np.empty((maxtrain,2),dtype=np.float32)
                flength = maxtrain
            else:
                frames = np.empty((framesleft,763002),dtype=np.float32)
                labels = np.empty((framesleft,2),dtype=np.float32)
                flength = framesleft

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
                print("model: ",str(modelnum)," video: ",vidnum," frame: ",frameindex, "i: ",i + 1," / ",totalf)
                image = cv2.resize(frame, (582,437))
                cnewframe = image.reshape((763002))
                frame = cnewframe.astype('float32')
                frame /= 255    

                frames[trainindex] = frame 

                labels[trainindex][0] = float(dataset0[index][0])
                labels[trainindex][1] = float(dataset0[index][1])

                trainindex += 1
                trainedframes += 1
                framesleft -= 1
                i += 1 
        
        model.fit(x=frames,y=labels,epochs=5,batch_size=50)
        trainedframes = 0

    #delete initals

    del frames
    del labels
    gc.collect()

    return model
    
def getresults(model,num,framlen,modelnum):
    cap0 = cv2.VideoCapture("unlabeled/" + str(num) + ".hevc")

    ret1, cframe = cap0.read()

    index = 0

    while index < framlen:
        sframes = 0
        print("model: ",str(modelnum)," video: ",str(num)," / 9" ," frame: ",index + 1, " / ",framlen)
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

def get_mse(gt, test):
  test = np.nan_to_num(test)
  return np.mean(np.nanmean((gt - test)**2, axis=0))

def eval(vpre):
    for i in range(0,5):
        gt = np.loadtxt(GT_DIR + str(i) + '.txt')
        zero_mses.append(get_mse(gt, np.zeros_like(gt)))

        test = np.loadtxt(TEST_DIR + str(i+5) + '.txt')
        print("test: ",len(test))
        print("gt: ",len(gt))
        mses.append(get_mse(gt, test))

    percent_err_vs_all_zeros = 100*np.mean(mses)/np.mean(zero_mses)
    print("prevmodel: ",)
    print(f'YOUR ERROR SCORE IS {percent_err_vs_all_zeros:.2f}% (lower is better)')

    return percent_err_vs_all_zeros



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

#26.6

#algo:
#train
#set if zero
#else
#multiply by number of previous models
#add model pred
#incrment num mods
#divide
#print and eval

#plotting data for diemensionality reduction 
"""
fig = plt.figure()
ax = plt.axes(projection='3d')

zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
"""
seed(datetime.now())

#these literally don't matter, for mental
#total frames is random
#flength is dependent on total frames

flength = 50
totalframes = 5000

number_models = 1
currmod = 1 
prevmod = 0
pet = 25.0
mer = 26.0
prevmer = 0.0
zero_mses = []
mses = []

TEST_DIR = 'unlabeled/'
GT_DIR = 'labeled/'


start = time.time()


labels5 = np.empty((1200,2),dtype=np.float32)
labels6 = np.empty((1200,2),dtype=np.float32)
labels7 = np.empty((1200,2),dtype=np.float32)
labels8 = np.empty((1200,2),dtype=np.float32)
labels9 = np.empty((1196,2),dtype=np.float32)


while mer >= pet:
    frame = np.empty((763002),dtype=np.float32)

    #randomizing
    totalframes = 1#randint(1,5000)

    model = getmod()
    model = train(model,totalframes,flength,number_models,totalframes)

    
    del frame
    gc.collect()

    prelabels = np.empty((1200,2),dtype=np.float32)
    testframe = np.empty((1,763002),dtype=np.float32)

    answers5 = getresults(model,5,len(labels5),number_models)
    answers6 = getresults(model,6,len(labels6),number_models)
    answers7 = getresults(model,7,len(labels7),number_models)
    answers8 = getresults(model,8,len(labels8),number_models)
    answers9 = getresults(model,9,len(labels9),number_models)


    for i in range(len(labels5)):
        if number_models == 1:
            labels5[i][0] = answers5[number_models-1][0]
            labels5[i][1] = answers5[number_models-1][1]
        else:
            #mult by prev
            labels5[i][0] = labels5[i][0] * prevmod
            labels5[i][1] = labels5[i][1] * prevmod

            #add
            labels5[i][0] += answers5[number_models-1][0]
            labels5[i][1] += answers5[number_models-1][1]

    for i in range(len(labels6)):
        if number_models == 1:
            labels6[i][0] = answers6[number_models-1][0]
            labels6[i][1] = answers6[number_models-1][1]
        else:
            #mult by prev
            labels6[i][0] = labels6[i][0] * prevmod
            labels6[i][1] = labels6[i][1] * prevmod

            #add
            labels6[i][0] += answers6[number_models-1][0]
            labels6[i][1] += answers6[number_models-1][1]

    for i in range(len(labels7)):
        if number_models == 1:
            labels7[i][0] = answers7[number_models-1][0]
            labels7[i][1] = answers7[number_models-1][1]
        else:
            #mult by prev
            labels7[i][0] = labels7[i][0] * prevmod
            labels7[i][1] = labels7[i][1] * prevmod

            #add
            labels7[i][0] += answers7[number_models-1][0]
            labels7[i][1] += answers7[number_models-1][1]

    for i in range(len(labels8)):
        if number_models == 1:
            labels8[i][0] = answers8[number_models-1][0]
            labels8[i][1] = answers8[number_models-1][1]
        else:
            #mult by prev
            labels8[i][0] = labels8[i][0] * prevmod
            labels8[i][1] = labels8[i][1] * prevmod

            #add
            labels8[i][0] += answers8[number_models-1][0]
            labels8[i][1] += answers8[number_models-1][1]

    for i in range(len(labels9)):
        if number_models == 1:
            labels9[i][0] = answers9[number_models-1][0]
            labels9[i][1] = answers9[number_models-1][1]
        else:
            #mult by prev
            labels9[i][0] = labels9[i][0] * prevmod
            labels9[i][1] = labels9[i][1] * prevmod

            #add
            labels9[i][0] += answers9[number_models-1][0]
            labels9[i][1] += answers9[number_models-1][1]

    del prelabels
    del testframe

    del answers5
    del answers6
    del answers7
    del answers8
    del answers9
    gc.collect()


    #divide by num of prev models + 1

    for x in range(len(labels5)):
        labels5[x][0] = (labels5[x][0] / number_models)
        labels5[x][1] = (labels5[x][1] / number_models)

    for x in range(len(labels6)):
        labels6[x][0] = (labels6[x][0] / number_models)
        labels6[x][1] = (labels6[x][1] / number_models)

    for x in range(len(labels7)):
        labels7[x][0] = (labels7[x][0] / number_models)
        labels7[x][1] = (labels7[x][1] / number_models)

    for x in range(len(labels8)):
        labels8[x][0] = (labels8[x][0] / number_models)
        labels8[x][1] = (labels8[x][1] / number_models)

    for x in range(len(labels9)):
        labels9[x][0] = (labels9[x][0] / number_models)
        labels9[x][1] = (labels9[x][1] / number_models)

    np.savetxt("unlabeled/5.txt", labels5)
    np.savetxt("unlabeled/6.txt", labels6)
    np.savetxt("unlabeled/7.txt", labels7)
    np.savetxt("unlabeled/8.txt", labels8)
    np.savetxt("unlabeled/9.txt", labels9)
    
    prevmer = mer
    mer = eval(prevmer)

    prevmod = number_models
    number_models += 1

finished = time.time()

print("success!")
print("time elasped: ",(finished - start)/60, " minutes or ",(finished - start)/3600," hours")
