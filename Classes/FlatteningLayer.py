from Classes.Layer import Layer
from Matrice import Matrice


class FlatteningLayer(Layer):
    def __init__(self):
        self.inputShape = (-1, -1)
        self.outputShape = (-1, -1)
        self.isFullInit = False

    def __repr__(self):
        return 'FLAT'

    def feedForward(self, inputs: list[Matrice]) -> list[Matrice]:
        if not self.isFullInit:
            self.inputShape = inputs[0].shape
            self.outputShape = (self.inputShape[0] * self.inputShape[1], 1)
            self.isFullInit = True

        return list(map(lambda x: x.reshape(self.outputShape), inputs))

    def backPropagation(self, outputGradients: list[Matrice], learningRate: float) -> list[Matrice]:
        return list(map(lambda x: x.reshape(self.inputShape), outputGradients))
