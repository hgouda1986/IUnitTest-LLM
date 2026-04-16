import torch.nn as nn

class LLMModule(nn.Module):
    def __init__(self, dim):
        super(LLMModule, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, 2)

    def forward(self, x):
        x = nn.functional.relu(self.fc(x))
        return self.out(x)