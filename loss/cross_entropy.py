from activations import softmax
from .base_loss import Loss
import numpy as np

class CrossEntropyWithLogits(Loss):
    def __init__(self, size_average=True):
        """
        - size_average: if dividing by the batch size or not
        - logit: intermediate variables to store the scores
        - label: Ground truth label for classification task
        """
        self.size_average = size_average
        self.logit = None
        self.label = None

    def forward(self, feat, label):
        logit = softmax(feat)
        loss = None
        one_hot = np.zeros(label.shape + (logit.shape[-1],))
        one_hot[np.arange(label.size), label] = 1
        loss = -1 * np.mean(np.sum(one_hot * np.log(logit), axis=-1))
        self.logit = logit
        self.label = label
        return loss

    def backward(self):
        logit = self.logit
        label = self.label
        if logit is None:
            raise ValueError("No forward function called before for this module!")
        dlogit = None
        one_hot = np.zeros(label.shape + (logit.shape[-1],))
        one_hot[np.arange(label.size), label] = 1
        dlogit = (logit - one_hot) / logit.shape[0]
        self.logit = None
        self.label = None
        return dlogit