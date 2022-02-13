from Classes.CNN import CNN
from Classes.ImageData import ImageData
from Matrice import Matrice

inputShape = (5, 5)

base = ImageData('Z:\\_DataSets\\TIPE\\CMFD\\0001.jpg')
image = base.getMatrice()
matrice = Matrice.random(15, 15, -5, 5, float)

# print(image.shape)
network = CNN(0.5)

network.addLayer(CNN.CONV, **{'kernelDim': 3, 'nbKernel': 5})  # dim output : (176, 162)
network.addLayer(CNN.ReLU, **{'typeReLU': 'max'})  # dim output : (176, 162)
network.addLayer(CNN.POOL, **{'typePooling': 'max', 'filterDim': 3})  # dim output : (88, 81)
network.addLayer(CNN.FLAT)
network.addLayer(CNN.FC, **{'outputShape': (5, 1)})
network.addLayer(CNN.LOSS, **{'typeLoss': 'bce'})

# start = time()
output = network.feedForward([matrice])
print(output[0])
network.backPropagation(output[0], Matrice.random(5, 1, -3, 3, float))
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
