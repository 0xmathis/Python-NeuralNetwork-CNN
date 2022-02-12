from Matrice import Matrice

MAX = 'max'
AVG = 'average'


class PoolingLayer:
    def __init__(self, typePooling: str, filterDim: int):
        if typePooling not in (MAX, AVG):
            raise ValueError

        self.typePooling = typePooling
        self.filterDim = filterDim
        self.inputShape = (-1, -1)
        self.outputShape = (-1, -1)
        self.isFullInit = False

        self.inputs = []
        self.outputs: list[Matrice] = []  # il y aura nbInputs outputs

    def __repr__(self):
        return f'POOL {self.typePooling} {self.filterDim}x{self.filterDim}'

    def feedForward(self, inputs: list[Matrice]) -> list[Matrice]:
        if not self.isFullInit:
            self.inputShape = inputs[0].shape
            self.outputShape = (self.inputShape[0] // self.filterDim, self.inputShape[1] // self.filterDim)
            self.isFullInit = True

        self.inputs = inputs.copy()

        self.outputs = [self.pooling(matrice) for matrice in self.inputs]

        return self.outputs

    def pooling(self, input_: Matrice) -> Matrice:
        output = Matrice.vide(input_.getRows() // self.filterDim, input_.getColumns() // self.filterDim)

        for i in range(output.getRows()):
            for j in range(output.getColumns()):
                subInput = input_.getSubMatrice((i * self.filterDim, j * self.filterDim), ((i + 1) * self.filterDim - 1, (j + 1) * self.filterDim - 1))

                if self.typePooling == MAX:
                    output[(i, j)] = max(subInput[(i, j)] for j in range(subInput.getColumns()) for i in range(subInput.getRows()))
                else:
                    output[(i, j)] = sum(subInput) / (subInput.getRows() * subInput.getColumns())

        return output

    def backPropagation(self):
