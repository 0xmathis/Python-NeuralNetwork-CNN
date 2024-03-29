from math import cosh, tanh
from sympy import exp, N

from Classes.Layer import Layer
from Matrice import Matrice

MAX = 'max'
SIGMOID = 'sigmoid'
TANH = 'tanh'
STEP = 'step'


class ReluLayer(Layer):
    def __init__(self, typeReLU: str):
        if typeReLU not in (MAX, SIGMOID, TANH, STEP):
            raise ValueError

        self.typeReLU: str = typeReLU
        self.inputs: list[Matrice] = []
        self.outputs: list[Matrice] = []

        if self.typeReLU == MAX:
            self.activation = lambda x: max(0, x)
            self.activationPrime = lambda y: 1 if y > 0 else 0
        elif self.typeReLU == SIGMOID:
            self.activation = lambda x: N(1 / (1 + exp(-x)))
            self.activationPrime = lambda y: N(self.activation(y) * (1 - self.activation(y)))
        elif self.typeReLU == TANH:
            self.activation = lambda x: tanh(x)
            self.activationPrime = lambda y: 1 / cosh(y) ** 2
        else:
            self.activation = lambda x: 1 if x > 0 else 0
            self.activationPrime = lambda y: 0

    def __repr__(self):
        return f'ReLU {self.typeReLU}'

    def feedForward(self, inputs: list[Matrice]) -> list[Matrice]:
        self.inputs = inputs.copy()
        self.outputs = [matrice.map(self.activation) for matrice in inputs]

        return self.outputs

    def backPropagation(self, outputGradients: list[Matrice], learningRate: float) -> list[Matrice]:
        return [self.inputs[i].map(self.activationPrime).hp(outputGradients[i]) for i in range(len(self.inputs))]
