from math import cosh, tanh

from sympy import exp, N

from Matrice import Matrice

MAX = 'max'
SIGMOID = 'sigmoid'
TANH = 'tanh'
STEP = 'step'


class ReluLayer:
    def __init__(self, typeReLU: str):
        self.typeReLU: str = typeReLU
        self.input_: list[Matrice] = []
        self.output: list[Matrice] = []

        if self.typeReLU == MAX:
            self.activation = lambda x: max(0, x)
            self.d_activation_dx = lambda y: 1 if y > 0 else 0
        elif self.typeReLU == SIGMOID:
            self.activation = lambda x: N(1 / (1 + exp(-x)))
            self.d_activation_dx = lambda y: N(self.activation(y) * (1 - self.activation(y)))
        elif self.typeReLU == TANH:
            self.activation = lambda x: tanh(x)
            self.d_activation_dx = lambda y: 1 / cosh(y) ** 2
        else:
            self.activation = lambda x: 1 if x > 0 else 0
            self.d_activation_dx = lambda y: 0

    def feedForward(self, input_: list[Matrice]) -> list[Matrice]:
        self.input_ = input_.copy()
        self.output = [matrice.map(self.activation) for matrice in input_]
        return self.output

    def backPropagation(self, outputGradients: list[Matrice], learningRate: float) -> list[Matrice]:
        return [self.input_[i].map(self.d_activation_dx).hp(outputGradients[i]) for i in range(len(self.input_))]
