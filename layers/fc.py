import numpy as np

class FC(object):
    def __init__(self, input_dim, output_dim, init_scale=0.002, name="fc"):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - input_dim: input dimension
        - output_dim: output dimension
        - meta: to store the forward pass activations for computing backpropagation
        """
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(input_dim, output_dim)
        self.params[self.b_name] = np.zeros(output_dim)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None

    def forward(self, feat):
        output = None
        assert (
            len(feat.shape) == 2 and feat.shape[-1] == self.input_dim
        ), "But got {} and {}".format(feat.shape, self.input_dim)
        output = np.matmul(feat, self.params[self.w_name]) + self.params[self.b_name]
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        assert (
            len(feat.shape) == 2 and feat.shape[-1] == self.input_dim
        ), "But got {} and {}".format(feat.shape, self.input_dim)
        assert (
            len(dprev.shape) == 2 and dprev.shape[-1] == self.output_dim
        ), "But got {} and {}".format(dprev.shape, self.output_dim)
        w = self.params[self.w_name]
        wdprev = np.matmul(w, np.expand_dims(dprev, axis=-1))
        dfeat = wdprev.reshape(wdprev.shape[:-1])
        self.grads[self.w_name] = np.dot(feat.T, dprev)
        self.grads[self.b_name] = np.sum(dprev, axis=0)
        self.meta = None
        return dfeat

