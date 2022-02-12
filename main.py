from time import time

from matplotlib.pyplot import imshow, show
from numpy import array

from Classes.CNN import CNN, CONV, POOL, ReLU
from Classes.ImageData import ImageData
from Matrice import Matrice

inputShape = (5, 5)

base = ImageData('Z:\\_DataSets\\TIPE\\CMFD\\0001.jpg')
image = base.getMatrice()
matrice = Matrice([[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 10],
                   [11, 12, 13, 14, 15],
                   [16, 17, 18, 19, 20],
                   [21, 22, 23, 24, 25]])

# print(image.shape)
network = CNN()

conv1 = {'kernelDim': 3, 'nbKernel': 5}
network.addLayer(CONV, **conv1)  # dim output : (176, 162)

relu1 = {'typeReLU': 'max'}
network.addLayer(ReLU, **relu1)  # dim output : (176, 162)

pool1 = {'typePooling': 'max', 'filterDim': 3}
network.addLayer(POOL, **pool1)  # dim output : (88, 81)

conv2 = {'kernelDim': 3, 'nbKernel': 5}
network.addLayer(CONV, **conv2)  # dim output : (172, 158)

relu2 = {'typeReLU': 'max'}
network.addLayer(ReLU, **relu2)  # dim output : (172, 158)

start = time()
output = network.feedForward([image])
print(time() - start)

# start = time()
# output2 = network.feedForward([image])
# print(time() - start)

# print(output[0].shape)
#
inputArray = array(image.toList(), float)
imshow(inputArray)
show()

for i in range(len(output)):
    outputArray = array(output[i].toList(), float)
    imshow(outputArray)
    show()
