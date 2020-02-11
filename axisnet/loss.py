"""
A loss function measures how good our predictions are.
It's used to adjust parameters in the network.
"""

import numpy as np

from axisnet.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError




class MSE(Loss):
    """
    MSE is the Mean Squared Error, but we're actually using total squared error
    """
    
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)


class SSE(Loss):
    pass