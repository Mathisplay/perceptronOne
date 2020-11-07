print("Hello")
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image as im
imageCount = 5

print("Generating random weights...")
weights = np.round(np.random.uniform(-0.005, 0.005, 6252500), 3).reshape(2500, 2501)

print("Searching for data...")
with open("trainingData2/data.txt", "r") as f:
    print("Data found")
    text = f.readlines()
    global splitText
    splitText = np.zeros((len(text), 2), dtype=object)
    for i in range(len(text)):
        splitText[i] = text[i].rstrip().split(" ")
print("Analyzing images...")
learningData = np.zeros((len(splitText), 2501), dtype='float32')
for i in range(len(splitText)):
    image = im.open('trainingData2/' + splitText[i][0])
    learningData[i] = np.r_[np.array(image.getdata()).flatten()[0::3] / 255.0, 1]
bestWeights = weights.copy()
bestLifetime = 0
learning = 0.001
lifetime = 0

for iter in range(100):
    ok = True
    if iter % 10 == 0:
        print(iter)
    current = iter % imageCount
    tempData = learningData[current].copy()
    for i in range(10):
        id = np.random.randint(2500)
        if tempData[id] > 0.0:
            tempData[id] = 0.0
        else:
            tempData[id] = 1.0
    ans = -1.0
    sum = np.zeros(2500, dtype='float32')
    for i in range(2500):
        sum[i] = np.sum(np.multiply(tempData, weights[i]))
        if sum[i] < 0.0:
            ans = -1.0
            sum[i] = 0.0
        else:
            ans = 1.0
            sum[i] = 1.0
        if tempData[i] != sum[i]:
            weights[i] = weights[i] + (learning * (tempData[i] - ans) * tempData)
            lifetime = 0
        else:
            lifetime += 1
            if lifetime > bestLifetime:
                bestWeights = weights.copy()
                bestLifetime = lifetime
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
data = np.r_[np.array(image.getdata()).flatten()[0::3] / 255.0, 1]

plt.subplot(1, 7, 6)
plt.grid(False)
plt.xticks([])
plt.yticks([])
showPic = data[0:-1]
plt.imshow(showPic.reshape(50, 50), cmap=plt.cm.binary)
sum = np.zeros(2501, dtype='float32')
sum[-1] = 1
print("This drawing could be a:")
for i in range(2500):
    sum[i] = np.sum(np.multiply(data, bestWeights[i]))
    if sum[i] >= 0:
        sum[i] = 1
    else:
        sum[i] = 0
plt.subplot(1, 7, 7)
plt.grid(False)
plt.xticks([])
plt.yticks([])
showPic = sum[0:-1]
plt.imshow(showPic.reshape(50, 50), cmap=plt.cm.binary)
plt.show()
