import matplotlib.pyplot as plt
import numpy as np

class BooleanFunctions():

    def __init__(self, val_list):
        self.argmunets = val_list
        self.threshold = 0


    def boolAND(self):
        """
        Boolean function AND returns 1 or 0.
        Parameters:
        ----------
        x: boolean
        y: boolean
        ----------
        returns boolean 1 or 0 depending on the threshold.
        """
        self.threshold = len(self.argmunets)
        if sum(self.argmunets) >= self.threshold:
            return 1, self.threshold
        return 0, self.threshold
        
    def boolOR(self):
        """
        Boolean function AND returns 1 or 0.
        Parameters:
        ----------
        x: boolean
        y: boolean
        ----------
        returns boolean 1 or 0 depending on the threshold.
        """
        self.threshold = 1
        if sum(self.argmunets) >= self.threshold:
            return 1, self.threshold
        return 0, self.threshold

    def notAND(self):
        self.argmunets = list(map(lambda x: int(not x), self.argmunets))
        val = self.boolOR()
        return val
    
    def notOR(self):
        self.argmunets = list(map(lambda x: int(not x), self.argmunets))
        val = self.boolAND()
        return val


class BoolPlots():

    def linePlot(self, threshold):
        x_bool = [0, 1, 0, 1]
        y_bool = [1, 0, 0, 1]

        fig = plt.figure()
        x = np.linspace(0.1, 1.1)
        y = np.linspace(0.1, 1.1)
        plt.plot(x, threshold-x, label=f'x+y-{threshold}=0')
        plt.legend()
        plt.scatter(x_bool, y_bool, c='r')
        return fig