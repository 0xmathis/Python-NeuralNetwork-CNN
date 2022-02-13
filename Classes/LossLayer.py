from math import log

from Matrice import Matrice

BCE = 'bce'
MSE = 'mse'


class LossLayer:
    def __init__(self, typeLoss: str, targets: Matrice):
        if typeLoss not in (BCE, MSE):
            raise ValueError

        self.typeLoss = typeLoss

        if self.typeLoss == BCE:
            self.loss = self.BCE
            self.lossPrime = self.BCEprime
        else:
            self.loss = self.MSE
            self.lossPrime = self.MSEprime

    @staticmethod
    def BCE(outputs: Matrice, targets: Matrice) -> float:
        return -sum(targets.hp(outputs.map(log)) + (1 - targets).hp((1 - outputs).map(log))) / targets.getRows()

    @staticmethod
    def BCEprime(outputs: Matrice, targets: Matrice) -> Matrice:
        return ((1 - targets) / (1 - outputs) - targets / outputs) / targets.getRows()

    @staticmethod
    def MSE(outputs: Matrice, targets: Matrice):
        return (targets - outputs).ps(0, 0) / targets.getRows()

    @staticmethod
    def MSEprime(outputs: Matrice, targets: Matrice):
        return (outputs - targets) * 2 / targets.getRows()
