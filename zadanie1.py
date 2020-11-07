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

print("Searching for data...")
with open("trainingData/data.txt", "r") as f:
    print("Data found")
    text = f.readlines()
    global splitText
    splitText = np.zeros((len(text), 2), dtype=object)
    for i in range(len(text)):
        splitText[i] = text[i].rstrip().split(" ")
learningData = np.zeros((len(splitText), 36), dtype='float32')
correctAnswers = np.zeros(len(splitText), dtype='int')
print("Analyzing images...")
for i in range(len(splitText)):
    image = im.open('trainingData/' + splitText[i][0])
    learningData[i] = np.r_[np.array(image.getdata()).flatten()[0::3] / 255.0, 1]
    correctAnswers[i] = int(splitText[i][1])
bestWeights = weights.copy()
bestLifetime = 0
learning = 0.001
lifetime = 0

for i in range(10000):
    current = np.random.randint(len(splitText))
    correctAns = correctAnswers[current]
    noise = np.random.randint(0, 35, (2))
    learningDataNoise = learningData[current].copy()
    for a in noise:
        if learningDataNoise[a] > 0.0:
            learningDataNoise[a] = 0.0
        else:
            learningDataNoise[a] = 1.0
    sum = np.zeros(10, dtype='float32')
    for i in range(10):
        sum[i] = np.sum(np.multiply(learningDataNoise, weights[i]))
    best = np.amax(sum)
    bestId = np.unravel_index(np.argmax(sum, axis=None), sum.shape)[0]
    if bestId != correctAns:
        weights[bestId] = weights[bestId] + (-1 * learning * learningDataNoise)
        weights[correctAns] = weights[correctAns] + (learning * learningDataNoise)
        lifetime = 0
    else:
        lifetime += 1
        if lifetime > bestLifetime:
            bestWeights = weights.copy()
            bestLifetime = lifetime
print("Images analyzed")

image = im.open('trainingData/test.png')
data = np.r_[np.array(image.getdata()).flatten()[0::3] / 255.0, 1]
testCorrect = np.array([6])

sum = np.zeros(10, dtype='float32')
print("This drawing could be a:")
for i in range(10):
    sum[i] = np.sum(np.multiply(data, bestWeights[i]))
print(np.unravel_index(np.argmax(sum, axis=None), sum.shape)[0])

plt.figure()
plt.grid(False)
plt.xticks([])
plt.yticks([])
showPic = data[0:-1]
plt.imshow(showPic.reshape(7, 5), cmap=plt.cm.binary)
plt.show()
