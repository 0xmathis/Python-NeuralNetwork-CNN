from typing import Union
from time import time

from Classes.ConvolutionalLayer import ConvolutionalLayer
from Classes.FC import FC as FcLayer
from Classes.LossLayer import LossLayer
from Classes.PoolingLayer import PoolingLayer
from Classes.ReluLayer import ReluLayer
from Classes.ReshapeLayer import ReshapeLayer
from Matrice import Matrice

# Type de layer :
CONV = 'conv'
POOL = 'pool'
ReLU = 'relu'
FC = 'fc'
LOSS = 'loss'
LAYERS = {
    CONV: ConvolutionalLayer,
    POOL: PoolingLayer,
    ReLU: ReluLayer,
    FC: FcLayer,
    LOSS: LossLayer
}


class CNN:
    def __init__(self):
        # self.inputShape = inputShape
        # self.outputShape = outputShape

        self.network: list[Union[ConvolutionalLayer, PoolingLayer, LossLayer, ReluLayer, ReshapeLayer, FcLayer]] = []

    def addLayer(self, layer: str, **args) -> None:
        if layer not in LAYERS.keys():
            raise ValueError

        self.network += [LAYERS[layer](**args)]

    def feedForward(self, data: list[Matrice]) -> list[Matrice]:
        for layer in self.network:
            start = time()
            data = layer.feedForward(data)
            print(layer, time() - start)

        return data
