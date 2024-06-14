import numpy as np
from .base_optimizer import Optimizer

class Adam(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def update(self, layer):
        for n,dv in layer.grads.items():
            if n not in self.mt:
                self.mt[n] = 0
                self.vt[n] = 0
            self.t += 1
            self.mt[n] = self.beta1 * self.mt[n] + (1 - self.beta1) * dv
            dv_square = np.square(dv)
            self.vt[n] = self.beta2 * self.vt[n] + (1 - self.beta2) * dv_square
            adjusted_momentum = self.mt[n] / (1 - np.power(self.beta1, self.t))
            adjusted_vt = self.vt[n] / (1 - np.power(self.beta2, self.t))
            layer.params[n] -= (self.lr * adjusted_momentum) / (np.sqrt(adjusted_vt) + self.eps)
