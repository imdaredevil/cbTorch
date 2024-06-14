from .base_optimizer import Optimizer
import numpy as np

class RMSPropOptimizer(Optimizer):
    def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
        self.net = net
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}  # decaying average of past squared gradients

    def update(self, layer):
        for n,dv in layer.grads.items():
            if n not in self.cache:
                self.cache[n] = 0
            dv_square = np.square(dv)
            self.cache[n] = self.decay * self.cache[n] + (1 - self.decay) * dv_square
            layer.params[n] -= (self.lr * dv) / np.sqrt(self.cache[n] + self.eps)