from typing import Union

from Classes.ConvolutionalLayer import ConvolutionalLayer
from Classes.FC import FC as FcLayer
from Classes.FlatteningLayer import FlatteningLayer
from Classes.LossLayer import LossLayer
from Classes.PoolingLayer import PoolingLayer
from Classes.ReluLayer import ReluLayer
from Matrice import Matrice

# Type de layer :
CONV = 'conv'
POOL = 'pool'
ReLU = 'relu'
FC = 'fc'
LOSS = 'loss'
FLAT = 'flat'
LAYERS = {
    CONV: ConvolutionalLayer,
    POOL: PoolingLayer,
    ReLU: ReluLayer,
    FC: FcLayer,
    LOSS: LossLayer,
    FLAT: FlatteningLayer
}


class CNN:
    def __init__(self):
        self.network: list[Union[ConvolutionalLayer, PoolingLayer, LossLayer, ReluLayer, FlatteningLayer, FcLayer]] = []

    def addLayer(self, layer: str, **args) -> None:
        if layer not in LAYERS.keys():
            raise ValueError

        self.network += [LAYERS[layer](**args)]

    def feedForward(self, data: list[Matrice]) -> list[Matrice]:
        for layer in self.network:
            data = layer.feedForward(data)

        return data
