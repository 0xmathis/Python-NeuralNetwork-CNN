from Matrice import Matrice


class ReshapeLayer:
    def __init__(self, inputShape: tuple[int, int], outputShape: tuple[int, int]):
        self.inputShape = inputShape
        self.outputShape = outputShape

    def feedFroward(self, input_: list[Matrice]) -> list[Matrice]:
        return list(map(lambda x: x.reshape(self.outputShape), input_))

    def backPropagation(self, outputGradient: list[Matrice]) -> list[Matrice]:
        return list(map(lambda x: x.reshape(self.inputShape), outputGradient))
