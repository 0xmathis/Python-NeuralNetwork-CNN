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
        return f'CONV {self.nbKernel} kernels {self.kernelDim}x{self.kernelDim}'

    def initFull(self, inputs: list[Matrice]) -> None:
        self.inputShape = inputs[0].shape
        self.inputDepth = len(inputs)
        self.outputShape = (self.inputShape[0] - self.kernelDim + 1, self.inputShape[1] - self.kernelDim + 1)
        self.kernels = [[Matrice.random(self.kernelDim, self.kernelDim, -3, 3, float) for _ in range(self.inputDepth)] for _ in range(self.nbKernel)]
        # self.kernels = [[Matrice.random(self.kernelDim, self.kernelDim, 0, 5, int) for _ in range(self.inputDepth)] for _ in range(self.nbKernel)]
        self.biases: list[Matrice] = [Matrice.random(self.outputShape[0], self.outputShape[1], -1, 1, float) for _ in range(self.nbKernel)]
        # self.biases: list[Matrice] = [Matrice.random(self.outputShape[0], self.outputShape[1], 0, 5, int) for _ in range(self.nbKernel)]
        # self.biases: list[Matrice] = [Matrice.full(self.outputShape[0], self.outputShape[1], 0) for _ in range(self.nbKernel)]
        # print('kernels:', len(self.kernels), len(self.kernels[0]), self.kernelDim, self.kernelDim, '\n')
        # for i in range(len(self.kernels)):
        #     for j in range(len(self.kernels[i])):
        #         print(i, j, '\n', self.kernels[i][j])
        # print('biases:', len(self.biases), 1, *self.outputShape, '\n')
        # for i in range(len(self.biases)):
        #     print(i, '\n', self.biases[i])

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
        kernelGradient: list[list[Matrice]] = [[Matrice.vide(1, 1) for _ in range(self.inputDepth)] for _ in range(self.nbKernel)]
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

    @staticmethod
    def validCorrelation(input_: Matrice, kernel: Matrice) -> Matrice:
        output = Matrice.vide(input_.getRows() - kernel.getRows() + 1, input_.getColumns() - kernel.getColumns() + 1)

        for i in range(output.getRows()):
            for j in range(output.getColumns()):
                subInput = input_.getSubMatrice((i, j), (i + kernel.getRows() - 1, j + kernel.getColumns() - 1))
                output[(i, j)] = sum(subInput.hp(kernel))

        return output

    @staticmethod
    def fullCorrelation(input_: Matrice, kernel: Matrice) -> Matrice:
        output = Matrice.vide(input_.getRows() + kernel.getRows() - 1, input_.getColumns() + kernel.getColumns() - 1)
        inputExpand = input_.copy()

        for _ in range(kernel.getRows() - 1):
            inputExpand = inputExpand.addNullRow(TOP)
            inputExpand = inputExpand.addNullRow(BOTTOM)
            inputExpand = inputExpand.addNullColumn(LEFT)
            inputExpand = inputExpand.addNullColumn(RIGHT)

        for i in range(output.getRows()):
            for j in range(output.getColumns()):
                subInput = inputExpand.getSubMatrice((i, j), (i + kernel.getRows() - 1, j + kernel.getColumns() - 1))
                output[(i, j)] = sum(subInput.hp(kernel))

        return output

    @staticmethod
    def rotation180Matrice(matrice: Matrice) -> Matrice:
        rotMatrice = Matrice([[1 if (i + j == matrice.getRows() - 1) else 0 for j in range(matrice.getColumns())] for i in range(matrice.getRows())])
        return rotMatrice * matrice * rotMatrice


conv = ConvolutionalLayer(kernelDim=2, nbKernel=2)

input_ = [Matrice.random(150, 150, 0, 5, int) for _ in range(3)]
from time import time

start = time()
print('output:\n', conv.feedForward(input_))
print(time() - start)
