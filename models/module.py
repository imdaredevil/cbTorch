class Module(object):
    def __init__(self):
        self.params = {}
        self.grads = {}
    
    def forward(*args, **kwargs):
        raise NotImplementedError

    def backward(*args, **kwargs):
        raise NotImplementedError

