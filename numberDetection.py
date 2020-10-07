from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image as im
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

with open("trainingData/data.txt", "r") as f:
    text = f.readlines()
    global splitText
    splitText = np.zeros((len(text), 2), dtype=object)
    for i in range(len(text)):
        splitText[i] = text[i].rstrip().split(" ")
finalData = np.zeros((len(text), 7, 5, 3), dtype='float32')
for i in range(len(splitText)):
    image = im.open('trainingData/' + splitText[i][0])
    data = np.array(image.getdata())
    data = data.reshape((7, 5, 3))
    finalData[i] = data.copy()
learningData = np.zeros((len(splitText), 7, 5), dtype='float32')
for i in range(len(finalData)):
    for j in range(7):
        for k in range(5):
            learningData[i][j][k] = finalData[i][j][k][0] / 255.0
plt.imshow(learningData[3, :, :])
#plt.figure()
#plt.imshow(x_train[2])
plt.show()
