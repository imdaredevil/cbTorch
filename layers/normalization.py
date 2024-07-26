import numpy as np
class LayerNormalization(object):
    def __init__(self, name="normalization"):
        """
        - name: the name of the layer (string)
        """
        self.name = name
        self.params = {}
        self.grads = {}
        self.stddev = None

    def forward(self, feat):
        mean = np.mean(feat, axis=-1)
        stddev = np.std(feat, axis = -1)
        output = ((feat - mean) / stddev).reshape(mean.size, -1 )
        self.stddev = stddev
        return output

    def backward(self, dprev):
        return dprev / self.stddev
