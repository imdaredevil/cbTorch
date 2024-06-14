from models import Module, Sequential
from layers import Flatten, FC
from loss import Loss, CrossEntropyWithLogits
from activations import LeakyRelu
from optimizers import Optimizer, Adam
import numpy as np
import tqdm

class FullyConnectedNetwork(Module):
    def __init__(self):
        """ Some comments """
        self.net = Sequential(
            Flatten(name="flat"),
            FC(2, 100, 5e-2, name="fc1"),
            LeakyRelu(name="relu1"),
            FC(100, 100, 5e-2, name="fc2"),
            LeakyRelu(name="relu2"),
            FC(100, 100, 5e-2, name="fc3"),
            LeakyRelu(name="relu3"),
            FC(100, 100, 5e-2, name="fc4"),
            LeakyRelu(name="relu4"),
            FC(100, 100, 5e-2, name="fc5"),
            LeakyRelu(name="relu5"),
            FC(100, 2, 5e-2, name="fc6")
        )

def train(x, y, model: Module, loss: Loss, optimizer: Optimizer, epoch):
    pbar = tqdm.tqdm(range(epoch))
    for _ in pbar:
        y_hat = model.forward(x)
        curr_loss = loss.forward(y_hat, y)
        pbar.set_description(f"Loss: {curr_loss}")
        dloss = loss.backward()
        model.backward(dloss)
        optimizer.step()

num_records = 1000
x = np.concatenate([
    np.zeros(shape = (num_records, 2)),
    np.ones(shape = (num_records, 2)),
    np.repeat([[0, 1]], num_records, axis=0),
    np.repeat([[1, 0]], num_records, axis=0)
], axis=0)
np.random.shuffle(x)

y = ((x[:, 0] + x[:, 1]) == 1)
y = y.astype(int)




model = FullyConnectedNetwork()
optim =  Adam(model.net)
loss = CrossEntropyWithLogits(True)

train(x, y, model, loss, optim, 100)

test_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
logits = model.forward(test_x)
print(logits.argmax(axis=1))

    
    