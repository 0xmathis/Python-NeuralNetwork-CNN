from Classes.CNN import CNN
from Classes.ImageData import ImageData
from Matrice import Matrice

inputShape = (5, 5)

base = ImageData('Z:\\_DataSets\\TIPE\\CMFD\\0001.jpg')
image = base.getMatrice()
matrice = Matrice.random(15, 15, -5, 5, float)

# print(image.shape)
network = CNN(0.5)

network.addLayer(CNN.CONV, **{'kernelDim': 5, 'nbKernel': 6})
network.addLayer(CNN.ReLU, **{'typeReLU': 'max'})
network.addLayer(CNN.POOL, **{'typePooling': 'max', 'filterDim': 4})

network.addLayer(CNN.CONV, **{'kernelDim': 5, 'nbKernel': 16})
network.addLayer(CNN.ReLU, **{'typeReLU': 'max'})
network.addLayer(CNN.POOL, **{'typePooling': 'max', 'filterDim': 4})

network.addLayer(CNN.FLAT)

network.addLayer(CNN.FC, **{'outputShape': (100, 1)})
network.addLayer(CNN.ReLU, **{'typeReLU': 'sigmoid'})

network.addLayer(CNN.FC, **{'outputShape': (2, 1)})
network.addLayer(CNN.ReLU, **{'typeReLU': 'sigmoid'})

network.addLayer(CNN.LOSS, **{'typeLoss': 'bce'})


def train(nbIteration: int) -> float:
    time: float = 0

    for i in range(nbIteration):
        input: Matrice = Matrice.random(180, 166, -5, 5, float)
        target: Matrice = Matrice.random(2, 1, -3, 3, float)

        time += network.trainFromExternalData(input, target, i + 1)

    return time / nbIteration


train(1)
train(100)
