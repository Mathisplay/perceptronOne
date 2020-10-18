print("Hello")
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image as im
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import tensorflow as tf
#from tensorflow import keras
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
print("Generating random weights...")
weights = np.round(np.random.uniform(-0.001, 0.001, 360), 3).reshape(10, 36)
weights = np.round(weights / 1000.0, 3)
#print(weights[0][0])
#print(np.sum(weights))
#print(theta)
print("Searching for data...")
with open("trainingData/data.txt", "r") as f:
    print("Data found")
    text = f.readlines()
    global splitText
    splitText = np.zeros((len(text), 2), dtype=object)
    for i in range(len(text)):
        splitText[i] = text[i].rstrip().split(" ")
finalData = np.zeros((len(text), 7, 5, 3), dtype='float32')
print("Analyzing images...")
for i in range(len(splitText)):
    image = im.open('trainingData/' + splitText[i][0])
    data = np.array(image.getdata())
    data = data.reshape((7, 5, 3))
    finalData[i] = data.copy()
learningData = np.zeros((len(splitText), 36), dtype='float32')
correctAnswers = np.zeros(len(splitText), dtype='int')
for i in range(len(finalData)):
    for j in range(7):
        for k in range(5):
            learningData[i][j * 5 + k] = finalData[i][j][k][0].copy() / 255.0
    correctAnswers[i] = int(splitText[i][1])
    learningData[i][-1] = 1

bestWeights = weights.copy()
bestLifetime = 0
learning = 0.001
lifetime = 0
while bestLifetime <= 10000:
    bestDetection = -1

    current = np.random.randint(len(splitText))
    correctAns = correctAnswers[current]
    sum = np.zeros(10, dtype='float32')
    for i in range(10):
        sum[i] = np.sum(np.multiply(learningData[current], weights[i]))
    best = np.amax(sum)
    bestId = np.unravel_index(np.argmax(sum, axis=None), sum.shape)[0]

    if bestId != correctAns:
        weights[bestId] = weights[bestId] + (learning * (-2.0) * learningData[current])
        weights[correctAns] = weights[correctAns] + (learning * (2.0) * learningData[current])
        lifetime = 0
    else:
        lifetime += 1
        if lifetime > bestLifetime:
            bestWeights = weights.copy()
            bestLifetime = lifetime
print("Images analyzed")

image = im.open('trainingData/test.png')
data = np.array(image.getdata())
data = data.reshape((7, 5, 3))
newData = np.zeros((36), dtype='float32')
for i in range(7):
    for j in range(5):
        newData[i * 5 + j] = data[i][j][0].copy() / 255.0
newData[-1] = 1
testCorrect = np.array([6])

sum = np.zeros(10, dtype='float32')
print("This drawing could be a:")
for i in range(10):
    sum[i] = np.sum(np.multiply(newData, bestWeights[i]))
print(np.unravel_index(np.argmax(sum, axis=None), sum.shape)[0])

plt.figure()
plt.grid(False)
plt.xticks([])
plt.yticks([])
showPic = newData[0:-1]
plt.imshow(showPic.reshape(7, 5), cmap=plt.cm.binary)
plt.show()
