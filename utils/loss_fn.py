import torch.nn as nn


class CE(nn.Module):
    """
    Traditional CrossEntropyLoss
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(**kwargs)

    def forward(self, X, y):
        return self.loss_fn(X, y)
