print("Hello")
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image as im
imageCount = 5
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import tensorflow as tf
#from tensorflow import keras
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
print("Generating random weights...")
weights = np.round(np.random.uniform(-0.005, 0.005, 50 * 50 * (50 * 50 + 1)), 3).reshape(50 * 50, 50 * 50 + 1)
weights = np.round(weights / 1000.0, 3)
#print(weights[0][0])
#print(np.sum(weights))
#print(theta)
print("Searching for data...")
with open("trainingData2/data.txt", "r") as f:
    print("Data found")
    text = f.readlines()
    global splitText
    splitText = np.zeros((len(text), 2), dtype=object)
    for i in range(len(text)):
        splitText[i] = text[i].rstrip().split(" ")
finalData = np.zeros((len(text), 50, 50, 3), dtype='float32')
print("Analyzing images...")
for i in range(len(splitText)):
    image = im.open('trainingData2/' + splitText[i][0])
    data = np.array(image.getdata())
    data = data.reshape((50, 50, 3))
    finalData[i] = data.copy()
learningData = np.zeros((imageCount, 50 * 50 + 1), dtype='float32')
for i in range(len(finalData)):
    for j in range(50):
        for k in range(50):
            learningData[i][j * 50 + k] = finalData[i][j][k][0].copy() / 255.0
    learningData[i][-1] = 1

bestWeights = weights.copy()
bestLifetime = 0
learning = 0.001
lifetime = 0
while bestLifetime < 750:
    if bestLifetime % 100 == 0:
        print(bestLifetime)
    bestLifetime += 1
    ok = True
    current = bestLifetime % imageCount
    tempData = learningData[current].copy()
    for i in range(50):
        id = np.random.randint(2500)
        if tempData[id] == 1.0:
            tempData[id] = 0
        else:
            tempData[id] = 1
    ans = -1
    sum = np.zeros(50 * 50 + 1, dtype='float32')
    sum[-1] = 1
    for i in range(2500):
        sum[i] = np.sum(np.multiply(tempData, weights[i]))
        if sum[i] < 0:
            ans = -1
            sum[i] = 0
        else:
            ans = 1
            sum[i] = 1
        if int(learningData[current][i]) != ans:
#            print(learningData[current][i] != ans)
            weights[i] = weights[i] + (learning * (float(int(learningData[current][i]) - ans)) * tempData)
#            lifetime = 0
#            ok = False
#    if ok == True:
#        print(bestLifetime)
#        lifetime += 1
#        if lifetime > bestLifetime:
#            bestWeights = weights.copy()
#            bestLifetime = lifetime
bestWeights = weights.copy()
print("Images analyzed")

plt.figure(figsize=(7, 1))
plt.grid(False)
plt.xticks([])
plt.yticks([])

for i in range(5):
    plt.subplot(1, 7, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    showPic = learningData[i][:-1]
    plt.imshow(showPic.reshape(50, 50), cmap=plt.cm.binary)

image = im.open('trainingData2/test.png')
data = np.array(image.getdata())
data = data.reshape((50, 50, 3))
newData = np.zeros((50 * 50 + 1), dtype='float32')
for i in range(50):
    for j in range(50):
        newData[i * 50 + j] = data[i][j][0].copy() / 255.0
newData[-1] = 1
plt.subplot(1, 7, 6)
plt.grid(False)
plt.xticks([])
plt.yticks([])
showPic = newData[0:-1]
plt.imshow(showPic.reshape(50, 50), cmap=plt.cm.binary)
sum = np.zeros(50 * 50 + 1, dtype='float32')
sum[:-1] = 1
print("This drawing could be a:")
for iter in range(200):
    for i in range(50 * 50):
        sum[i] = np.sum(np.multiply(newData, bestWeights[i]))
        if sum[i] >= 0:
            sum[i] = 1
        else:
            sum[i] = 0
    newData = sum
plt.subplot(1, 7, 7)
plt.grid(False)
plt.xticks([])
plt.yticks([])
showPic = sum[0:-1]
plt.imshow(showPic.reshape(50, 50), cmap=plt.cm.binary)
plt.show()
