from time import time
from typing import Union

from Classes.ConvolutionalLayer import ConvolutionalLayer
from Classes.FcLayer import FcLayer
from Classes.FlatteningLayer import FlatteningLayer
from Classes.Layer import Layer
from Classes.LossLayer import LossLayer
from Classes.PoolingLayer import PoolingLayer
from Classes.ReluLayer import ReluLayer
from Matrice import Matrice
from Utile import from_secondes


class CNN:
    # Type de layer :
    CONV: str = 'conv'
    POOL: str = 'pool'
    ReLU: str = 'relu'
    FC: str = 'fc'
    FLAT: str = 'flat'
    LOSS: str = 'loss'
    LAYERS = {
        CONV: ConvolutionalLayer,
        POOL: PoolingLayer,
        ReLU: ReluLayer,
        FC: FcLayer,
        FLAT: FlatteningLayer,
        LOSS: LossLayer
        }

    def __init__(self, learningRate: float):
        self.network: list[Union[Layer, LossLayer]] = []
        self.learningRate = learningRate

    def addLayer(self, layer: str, **args) -> None:
        if layer not in CNN.LAYERS.keys():
            raise ValueError

        self.network += [CNN.LAYERS[layer](**args)]

    def feedForward(self, input_: Matrice) -> Matrice:
        print('FF', end=' ')
        data: list[Matrice] = [input_]

        for layer in self.network[:-1]:
            print(layer, end=' ')
            data = layer.feedForward(data)
        print()

        return data[0]

    def backPropagation(self, outputs: Matrice, targets: Matrice):
        print('BP', end=' ')
        gradient: list[Matrice] = self.network[-1].getGradient(outputs, targets)

        for layer in self.network[-2::-1]:
            print(layer, end=' ')
            gradient = layer.backPropagation(gradient, self.learningRate)

    def trainFromExternalData(self, input: Matrice, target: Matrice, iteration: int) -> float:
        start: float = time()

        output: Matrice = self.feedForward(input)
        print(time() - start)
        self.backPropagation(output, target)

        end: float = time()

        print(f'{iteration} :', from_secondes(end - start))
        return end - start
