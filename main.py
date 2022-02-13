from Classes.CNN import CNN, CONV, ReLU, POOL, FLAT
from Classes.ImageData import ImageData
from Matrice import Matrice

inputShape = (5, 5)

base = ImageData('Z:\\_DataSets\\TIPE\\CMFD\\0001.jpg')
image = base.getMatrice()
matrice = Matrice.random(15, 15, -5, 5, float)

# print(image.shape)
network = CNN(0.5)

conv1 = {'kernelDim': 3, 'nbKernel': 5}
network.addLayer(CONV, **conv1)  # dim output : (176, 162)

relu1 = {'typeReLU': 'max'}
network.addLayer(ReLU, **relu1)  # dim output : (176, 162)

pool1 = {'typePooling': 'max', 'filterDim': 3}
network.addLayer(POOL, **pool1)  # dim output : (88, 81)

network.addLayer(FLAT)

# fc1 = {'outputShape': (5, 1)}
# network.addLayer(FC, **fc1)
#
# loss1 = {'typeLoss': 'bce'}
# network.addLayer(LOSS, **loss1)
#
# start = time()
output = network.feedForward([matrice.toVector()])
print(output)
# print(time() - start)

# start = time()
# output2 = network.feedForward([image])
# print(time() - start)

# print(output[0].shape)
#
# inputArray = array(image.toList(), float)
# imshow(inputArray)
# show()
#
# for i in range(len(output)):
#     outputArray = array(output[i].toList(), float)
#     imshow(outputArray)
#     show()
