from Matrice import Matrice
from sympy import log


class LossLayer:  # Binary Cross Entropy Loss
    @staticmethod
    def binaryCrossEntropy(target: Matrice, output: Matrice) -> Matrice:
        oneVector = Matrice.full(target.getRows(), 1, 1)
        return -(1 / target.getRows()) * (target.hp(output.map(log)) + (oneVector - target).hp((oneVector - output).map(log)))

    @staticmethod
    def dbinaryCrossEntropy_dy(target: Matrice, output: Matrice) -> Matrice:
        oneVector = Matrice.full(target.getRows(), 1, 1)
        return (1 / target.getRows()) * ((oneVector - target).hp((oneVector * output).map(LossLayer.inverse)) - target.hp(output.map(LossLayer.inverse)))

    @staticmethod
    def inverse(x: float) -> float:
        return 1 / x
