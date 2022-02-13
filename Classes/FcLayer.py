from Layer import Layer
from Matrice import Matrice


class FcLayer(Layer):
    def __init__(self, outputShape: tuple[int, int]):
        self.inputShape: tuple[int, int] = (-1, -1)
        self.inputFlatShape: tuple[int, int] = (-1, -1)
        self.outputShape: tuple[int, int] = outputShape
        self.weights: Matrice = Matrice.vide(1, 1)
        self.biases: Matrice = Matrice.vide(1, 1)
        self.input: Matrice = Matrice.vide(1, 1)
        self.output: Matrice = Matrice.vide(1, 1)
        self.isFullInit = False

    def fullInit(self, inputs: list[Matrice]):
        self.inputShape: tuple[int, int] = (len(inputs), inputs[0].getRows() * inputs[0].getColumns())
        self.inputFlatShape: tuple[int, int] = (len(inputs) * inputs[0].getRows() * inputs[0].getColumns(), 1)
        self.weights: Matrice = Matrice.random(self.outputShape[0], self.inputShape[0], -1, 1, float)
        self.biases: Matrice = Matrice.random(self.outputShape[0], 1, -1, 1, float)

    def feedForward(self, inputs: list[Matrice]) -> Matrice:
        if not self.isFullInit:
            self.fullInit(inputs)
            self.isFullInit = True

        self.input = self.reshapeList(inputs)
        self.output = self.weights * self.input + self.biases

        return self.output

    def backPropagation(self, outputGradients: Matrice, learningRate: float) -> list[Matrice]:
        weightsGradient = outputGradients * self.input.T
        self.weights -= learningRate * weightsGradient
        self.biases -= learningRate * outputGradients

        return self.reshapeMatrice(self.weights.T * outputGradients)  # inputGradients

    @staticmethod
    def reshapeList(inputs: list[Matrice]) -> Matrice:
        """
        :param inputs: list de n matrices de shape (r, c)
        :return: matrice de shape (n * r * c, 1)
        """
        output = Matrice.vide(len(inputs) * inputs[0].getRows() * inputs[0].getColumns(), 1)
        k = 0
        for matrice in inputs:
            # print(matrice)
            for i in range(matrice.getRows()):
                # print(i)
                output[k, 0] = matrice[i, 0]
                k += 1

        return output

    def reshapeMatrice(self, input_: Matrice) -> list[Matrice]:
        """
        :param input_: matrice de shape (n * r * c, 1)
        :return: list de n matrices de shape (r, c)
        """
        pass


liste = [Matrice([[1, 2, 3], [4, 5, 6]]).toVector(), Matrice([[7, 8, 9], [10, 11, 12]]).toVector()]
reshapedListe = FcLayer.reshapeList(liste)
print(reshapedListe)