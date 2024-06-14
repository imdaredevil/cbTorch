import numpy as np

class Dropout(object):
    def __init__(self, keep_prob, seed=None, name="dropout"):
        """
        - name: the name of current layer
        - keep_prob: probability that each element is kept.
        - meta: to store the forward pass activations for computing backpropagation
        - kept: the mask for dropping out the neurons
        - is_training: dropout behaves differently during training and testing, use
                       this to indicate which phase is the current one
        - rng: numpy random number generator using the given seed
        Note: params and grads should be just empty dicts here, do not update them
        """
        self.name = name
        self.params = {}
        self.grads = {}
        self.keep_prob = keep_prob
        self.meta = None
        self.kept = None
        self.is_training = False
        self.rng = np.random.RandomState(seed)
        assert keep_prob >= 0 and keep_prob <= 1, "Keep Prob = {} is not within [0, 1]".format(keep_prob)

    def forward(self, feat, is_training=True, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        kept = None
        output = None
        if is_training and (self.keep_prob > 0):
            input_size = feat.size // feat.shape[0]
            selected_nodes = self.rng.choice(input_size, int(np.round(self.keep_prob * input_size)), replace=False)
            mask = np.zeros((input_size,))
            mask[selected_nodes] = 1
            mask = mask.reshape(feat.shape[1:])
            kept = mask
            output = (feat * kept) / self.keep_prob
        else:
            output = feat
            kept = np.ones(feat.shape[1:])
        self.kept = kept
        self.is_training = is_training
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        dfeat = None
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        if self.is_training and (self.keep_prob > 0):
            dfeat = (dprev * self.kept) / self.keep_prob
        else:
            dfeat = dprev
        self.is_training = False
        self.meta = None
        return dfeat