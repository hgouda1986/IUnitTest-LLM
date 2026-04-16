import torch
import torch.nn as nn

class SimpleGNN(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=128, dropout=0.4):
        super(SimpleGNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = torch.relu(self.linear(x))
        return self.dropout(x)