from Classes.Layer import Layer
from Matrice import Matrice


class FcLayer(Layer):
    def __init__(self, outputShape: tuple[int, int]):
        self.inputShape: tuple[int, int] = (-1, -1)
        self.inputFlatShape: tuple[int, int] = (-1, -1)
        self.outputShape: tuple[int, int] = outputShape
        self.biases: Matrice = Matrice.vide(1, 1)
        self.weights: Matrice = Matrice.vide(1, 1)
        self.input: Matrice = Matrice.vide(1, 1)
        self.output: Matrice = Matrice.vide(1, 1)
        self.isFullInit = False

    def __repr__(self):
        return f'FC {self.outputShape[0]} outputs'

    def fullInit(self, inputs: list[Matrice]):
        self.inputShape: tuple[int, int] = (len(inputs), inputs[0].getRows() * inputs[0].getColumns())
        self.inputFlatShape: tuple[int, int] = (len(inputs) * inputs[0].getRows() * inputs[0].getColumns(), 1)
        self.biases: Matrice = Matrice.random(self.outputShape[0], 1, -1, 1, float)
        self.weights: Matrice = Matrice.random(self.outputShape[0], self.inputFlatShape[0], -1, 1, float)

    def feedForward(self, inputs: list[Matrice]) -> list[Matrice]:
        if not self.isFullInit:
            self.fullInit(inputs)
            self.isFullInit = True

        self.input = self.reshapeList(inputs)
        self.output = self.weights * self.input + self.biases

        return [self.output]

    def backPropagation(self, outputGradients: list[Matrice], learningRate: float) -> list[Matrice]:
        weightsGradient = outputGradients[0] * self.input.T  # outputGradients ne contient que 1 valeur
        self.biases -= learningRate * outputGradients[0]
        self.weights -= learningRate * weightsGradient

        return self.reshapeMatrice(self.weights.T * outputGradients[0])  # inputGradients

    def reshapeList(self, inputs: list[Matrice]) -> Matrice:
        """
        :param inputs: list de n matrices de shape (r * c, 1)
        :return: matrice de shape (n * r * c, 1)
        """
        output = Matrice.vide(*self.inputFlatShape)
        k = 0
        for matrice in inputs:
            for i in range(matrice.getRows()):
                output[k, 0] = matrice[i, 0]
                k += 1

        return output

    def reshapeMatrice(self, input_: Matrice) -> list[Matrice]:
        """
        :param input_: matrice de shape (n * r * c, 1)
        :return: list de n matrices de shape (r, c)
        """

        return [input_.getSubMatrice((n * self.inputShape[1], 0), ((n + 1) * self.inputShape[1] - 1, 0)) for n in range(self.inputShape[0])]
