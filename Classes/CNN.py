from Classes.ConvolutionalLayer import ConvolutionalLayer
from Classes.FC import FC as FcLayer
from Classes.FlatteningLayer import FlatteningLayer
from Classes.Layer import Layer
from Classes.PoolingLayer import PoolingLayer
from Classes.ReluLayer import ReluLayer
from Matrice import Matrice

# Type de layer :
CONV = 'conv'
POOL = 'pool'
ReLU = 'relu'
FC = 'fc'
FLAT = 'flat'
LAYERS = {
    CONV: ConvolutionalLayer,
    POOL: PoolingLayer,
    ReLU: ReluLayer,
    FC: FcLayer,
    FLAT: FlatteningLayer
    }


class CNN:
    def __init__(self, learningRate: float):
        self.network: list[Layer] = []
        self.learningRate = learningRate

    def addLayer(self, layer: str, **args) -> None:
        if layer not in LAYERS.keys():
            raise ValueError

        self.network += [LAYERS[layer](**args)]

    def feedForward(self, data: list[Matrice]) -> list[Matrice]:
        for layer in self.network:
            data = layer.feedForward(data)

        return data
