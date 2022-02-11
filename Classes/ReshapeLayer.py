from Matrice import Matrice


class ReshapeLayer:
    def __init__(self, inputShape: tuple[int, int], outputShape: tuple[int, int]):
        self.inputShape = inputShape
        self.outputShape = outputShape

    def feedFroward(self, input_: Matrice) -> Matrice:
        return input_.reshape(self.outputShape)

    def backPropagation(self, outputGradient: Matrice) -> Matrice:
        return outputGradient.reshape(self.inputShape)
