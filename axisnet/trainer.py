"""
Helper class to train a neural network.
"""

from axisnet.tensor import Tensor
from axisnet.nn import NeuralNet
from axisnet.loss import Loss, MSE
from axisnet.optimizer import Optimizer, SGD
from axisnet.data import DataIterator, BatchIterator

def train(net: NeuralNet,
        inputs: Tensor,
        targets: Tensor,
        epochs: int = 5000,
        iterator: DataIterator = BatchIterator(),
        loss: Loss = MSE(),
        optimizer: Optimizer = SGD()) -> None:
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)    