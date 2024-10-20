import torch.nn as nn
import torch.optim as optim


def SGD(model, **kwargs):
    optimizer = optim.SGD(model.parameters(), **kwargs)
    return optimizer
