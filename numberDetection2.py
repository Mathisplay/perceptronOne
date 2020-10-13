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
weights = np.round(np.random.rand(10, 7, 5), 4)
weights.flatten()
theta = np.round(np.random.rand(10), 4)
#print(weights)
print(np.sum(weights))
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
learningData = np.zeros((len(splitText), 7, 5), dtype='float32')
correctAnswers = np.zeros(len(splitText), dtype='int')
for i in range(len(finalData)):
    for j in range(7):
        for k in range(5):
            learningData[i][j][k] = finalData[i][j][k][0].copy() / 255.0
    correctAnswers[i] = int(splitText[i][1])
learningData.flatten()
#print(learningData)
#print(correctAnswers)
for z in range(10):
    answers = np.full(10, -1)
    current = np.random.randint(len(splitText))
    sum = np.zeros(10, dtype='float32')
    for i in range(10):
        sum[i] = np.sum(np.multiply(learningData[current], weights[i]))
        if sum[i] < theta[i]:
            print("Is not a " + str(i))
        else:
            print("Is probably a " + str(i))
            if i != correctAnswers[current]:
                
        print(sum[i])


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
newData = np.zeros((1, 7, 5), dtype='float32')
for i in range(7):
    for j in range(5):
        newData[0][i][j] = data[i][j][0].copy() / 255.0

testCorrect = np.array([6])
