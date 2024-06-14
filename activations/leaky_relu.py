import numpy as np

class LeakyRelu(object):
    def __init__(self, negative_slope=0.01, name="leaky_relu"):
        """
        - negative_slope: value that negative inputs are multiplied by
        - name: the name of current layer
        - meta: to store the forward pass activations for computing backpropagation
        Note: params and grads should be just empty dicts here, do not update them
        """
        self.negative_slope = negative_slope
        self.name = name
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, feat):
        output = None
        c = self.negative_slope
        output = np.where(feat >= 0, feat, feat * c)
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat = None
        c = self.negative_slope
        dfeat = np.where(feat >= 0, dprev, c * dprev)
        self.meta = None
        return dfeat