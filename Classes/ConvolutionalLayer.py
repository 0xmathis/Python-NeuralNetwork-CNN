from Classes.Layer import Layer
from Matrice import *


class ConvolutionalLayer(Layer):
    def __init__(self, kernelDim: int, nbKernel: int):
        self.inputShape: tuple[int, int] = (-1, -1)
        self.outputShape: tuple[int, int] = (-1, -1)
        self.inputDepth: int = -1
        self.kernelDim: int = kernelDim
        self.nbKernel: int = nbKernel
        self.isFullInit: bool = False

        self.kernels: list[list[Matrice]] = []
        self.inputs: list[Matrice] = []  # liste de inputDepth matrices
        self.biases: list[Matrice] = []  # il y a autant de biais que de kernel
        self.outputs: list[Matrice] = []  # il y a autant d'output que de biais

    def __repr__(self):
        return f'CONV {self.nbKernel}kernels {self.kernelDim}x{self.kernelDim}'

    def initFull(self, inputs: list[Matrice]) -> None:
        self.inputShape = inputs[0].shape
        self.inputDepth = len(inputs)
        self.outputShape = (self.inputShape[0] - self.kernelDim + 1, self.inputShape[1] - self.kernelDim + 1)
        self.kernels = [[Matrice.random(self.kernelDim, self.kernelDim, -3, 3, float) for _ in range(self.inputDepth)] for _ in range(self.nbKernel)]
        self.biases: list[Matrice] = [Matrice.random(*self.outputShape, -1, 1, float) for _ in range(self.nbKernel)]

    def feedForward(self, inputs: list[Matrice]) -> list[Matrice]:
        if not self.isFullInit:
            self.initFull(inputs)
            self.isFullInit = True

        self.inputs = inputs.copy()
        self.outputs = self.biases.copy()

        for k in range(self.nbKernel):
            for l in range(self.inputDepth):
                self.outputs[k] += self.validCorrelation(self.inputs[l], self.kernels[k][l])

        return self.outputs

    def backPropagation(self, outputGradients: list[Matrice], learningRate: float) -> list[Matrice]:
        kernelGradient: list[list[Matrice]] = [[Matrice.vide(self.kernelDim, 1) for _ in range(self.nbKernel)] for _ in range(self.inputDepth)]
        inputGradient: list[Matrice] = [Matrice.vide(*self.inputShape) for _ in range(self.inputDepth)]

        for i in range(self.nbKernel):
            for j in range(self.inputDepth):
                kernelGradient[i][j] = self.validCorrelation(self.inputs[j], outputGradients[i])
                inputGradient[j] += self.fullCorrelation(outputGradients[i], self.rotation180Matrice(self.kernels[i][j]))

        for i in range(self.nbKernel):
            for j in range(self.inputDepth):
                self.kernels[i][j] -= learningRate * kernelGradient[i][j]
                self.biases[j] -= learningRate * outputGradients[j]

        return inputGradient

    def validCorrelation(self, input_: Matrice, kernel: Matrice) -> Matrice:
        output = Matrice.vide(input_.getRows() - self.kernelDim + 1, input_.getColumns() - self.kernelDim + 1)

        for i in range(output.getRows()):
            for j in range(output.getColumns()):
                subInput = input_.getSubMatrice((i, j), (i + self.kernelDim - 1, j + self.kernelDim - 1))
                output[(i, j)] = sum(subInput.hp(kernel))

        return output * (1 / sum(kernel))

    def fullCorrelation(self, input_: Matrice, kernel: Matrice) -> Matrice:
        output = Matrice.vide(self.inputShape[0] + self.kernelDim - 1, self.inputShape[1] + self.kernelDim - 1)
        inputExpand = input_.copy()

        for _ in range(self.kernelDim - 1):
            inputExpand = inputExpand.addNullRow(TOP)
            inputExpand = inputExpand.addNullRow(BOTTOM)
            inputExpand = inputExpand.addNullColumn(LEFT)
            inputExpand = inputExpand.addNullColumn(RIGHT)

        for i in range(output.getRows()):
            for j in range(output.getColumns()):
                subInput = inputExpand.getSubMatrice((i, j), (i + self.kernelDim - 1, j + self.kernelDim - 1))
                output[(i, j)] = sum(subInput.hp(kernel))

        return output

    @staticmethod
    def rotation180Matrice(matrice: Matrice) -> Matrice:
        rotMatrice = Matrice([[1 if (i + j == matrice.getRows() - 1) else 0 for j in range(matrice.getColumns())] for i in range(matrice.getRows())])
        return rotMatrice * matrice * rotMatrice

# CNN = ConvolutionalLayer((3, 3), 2, 1)
#
# matrice1 = Matrice([[1, 2, 3],
#                     [4, 5, 6],
#                     [7, 8, 9]])
#
# kernel = Matrice([[1, 1],
#                   [0, 0]])

# print(CNN.fullCorrelation(matrice1, ConvolutionalLayer.rotation180Matrice(kernel)))

# CNN.feedForward(matrice1)
#
# for k in range(1):
#     print(CNN.outputs[k], end='\n\n')

# print(ConvolutionalLayer.rotationMatrice(matrice1))
