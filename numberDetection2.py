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
weights = np.round(np.random.uniform(-1, 1, 360), 5).reshape(10, 36)
weights = np.round(weights / 1000.0, 5)
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
#print(learningData)
#print(correctAnswers)
bestWeights = weights.copy()
bestLifetime = 0
learning = 0.00001
loops = 0
lifetime = 1
while bestLifetime <= 20:
    loops += 1
    bestDetection = -1
#    if loops % 1 == 0:
#        print(str(loops) + " " + str(bestLifetime))
    current = np.random.randint(len(splitText))
    correctAns = correctAnswers[current]
#    print("Expected: " + str(correctAns) + "\nFound:")
    sum = np.zeros(10, dtype='float32')
    for i in range(10):
        sum[i] = np.sum(np.multiply(learningData[current], weights[i]))
        T = 0
        if i == correctAns:
            T = 1
        else:
            T = -1

        value = 0
        if sum[i] < 0:
            value = -1
        else:
            value = 1
#            print("\t" + str(i))

        if T - value != 0:
            weights[i] = weights[i] + (learning * (T - value) * learningData[current])
            lifetime = 0
        else:
            if lifetime > bestLifetime:
                bestWeights = weights.copy()
                bestLifetime = lifetime
    lifetime += 1


#fig = plt.figure(figsize=(4, 5))
#for i in range(3):
#    for j in range(10):
#        fig.add_subplot(3, 10, i * 10 + j + 1).set_title(str(correctAnswers[i * 10 + j]))
#        plt.xticks([])
#        plt.yticks([])
#        plt.grid(False)
#        plt.imshow(learningData[i * 10 + j], cmap=plt.cm.binary)
#plt.imshow(learningData[3, :, :])
#plt.figure()
#plt.imshow(x_train[2])
#plt.show()
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
#bestValue = -1000000
#bestValueId = -1
for i in range(10):
    sum = np.sum(np.multiply(learningData[current], bestWeights[i]))
#    if sum > bestValue:
#        bestValue = sum
#        bestValueId = i
    if sum >= 0:
        print(i)
