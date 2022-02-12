from abc import abstractmethod

from Matrice import Matrice


class Layer:
    @abstractmethod
    def feedForward(self, inputs: list[Matrice]) -> list[Matrice]:
        pass

    @abstractmethod
    def backPropagation(self, outputGradients: list[Matrice]) -> list[Matrice]:
        pass