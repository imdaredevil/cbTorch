import numpy as np
class Residual(object):
    """
    A wrapper that wraps any module to add residual connection
    """
    def __init__(self, mod):
        self.name = f"residual_{mod.name}"
        self.mod = mod
        self.params = {}
        self.grads = {}

    def forward(self, dinput):
        new_input = self.mod.forward(dinput) + dinput
        self.params = self.mod.params
        return new_input

    def backward(self, gradoutput):
        new_grad = self.mod.backward(gradoutput)
        self.grads = self.mod.grads
        return new_grad + np.ones(new_grad.shape, dtype = float)
