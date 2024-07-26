import numpy as np
from activations import softmax, softmax_backward

class ScaledDotProductAttention(object):
    def __init__(self):
        self.V = None
        self.Q = None
        self.K = None
        self.attention = None
        self.matmul = None
        
    def forward(self, input : np.ndarray):
        assert input.shape[-2] == 3, "The penultimate dimension must be three"
        # Unpack input into queries, keys and values
        query, key, value = input.take(0, -2), input.take(1, -2), input.take(2, -2)
        self.Q, self.K, self.V = query , key, value # cache for later use in backward pass (if needed)
        self.matmul = query @ key.T 
        self.attention = softmax(self.matmul / np.sqrt(query.shape[-1]))
        return self.attention * value
    
    def backward(self, output: np.ndarray):
        if self.V is None:
            raise ValueError("Model forward is not called")        
        grad_Q = self.V * self.K / np.sqrt(self.K.shape[-1])
        grad_K = self.V * self.Q.T / np.sqrt(self.K.shape[-1])
        grad_K *= output
        grad_Q *= output
        grad_V = self.attention * output
        grad = np.stack([grad_Q, grad_K , grad_V], axis = -1)
        return grad.moveaxis(-1, -2)
    