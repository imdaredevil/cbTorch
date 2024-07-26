import numpy as np
class Embedding(object):
    def __init__(self, num_embeddings, embedding_dim, name = "embedding"):
        self.name = name
        self.params = {
            f"w_{name}": np.random.randn(num_embeddings, embedding_dim), # num embeddings x dimension of each word vector (vocab) in the model
        }
        self.param_name = f"w_{name}"
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.inputs = None
        self.grads = {}
    
    def forward (self , inputs):
        assert len(inputs.shape) == 2
        self.inputs = inputs
        return self.params[self.param_name][inputs]

    def backward(self, dprev):
        if self.inputs is None:
            raise ValueError("Forward pass not happened")
        dprev_flat = dprev.reshape(-1, self.embedding_dim)
        input_flat = self.inputs.flatten()
        dw = np.zeros_like(self.params[self.param_name])
        np.add.at(dw, input_flat, dprev_flat)
        self.grads[self.param_name] = dw
        return None