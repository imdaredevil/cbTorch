from layers import Dropout

class Module(object):
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, feat, is_training=True, seed=None):
        output = feat
        for layer in self.net.layers:
            if isinstance(layer, Dropout):
                output = layer.forward(output, is_training, seed)
            else:
                output = layer.forward(output)
        self.net.gather_params()
        return output

    def backward(self, dprev):
        for layer in self.net.layers[::-1]:
            dprev = layer.backward(dprev)
        self.net.gather_grads()
        return dprev

