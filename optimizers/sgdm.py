from .base_optimizer import Optimizer

class SGDMomentumOptimizer(Optimizer):
    def __init__(self, net, lr=1e-4, momentum=0.0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # last update of the velocity

    def update(self, layer):
        for n,dv in layer.grads.items():
            if n not in self.velocity:
                self.velocity[n] = 0.0
            self.velocity[n] = self.momentum * self.velocity[n] - self.lr * dv
            layer.params[n] += self.velocity[n]
