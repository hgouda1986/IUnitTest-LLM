import torch.nn as nn

class QLoRALayer(nn.Module):
    def __init__(self, dim):
        super(QLoRALayer, self).__init__()
        self.A = nn.Linear(dim, 16)
        self.B = nn.Linear(16, dim)

    def forward(self, x):
        return x + self.B(self.A(x))