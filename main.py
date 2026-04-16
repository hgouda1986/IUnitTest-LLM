import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.model_selection import KFold

from model.gnn import SimpleGNN
from model.qlora import QLoRALayer
from model.llm_module import LLMModule

from utils.preprocessing import preprocess, build_vocab, encode_text
from utils.evaluation import evaluate
from data.dataset_loader import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = SimpleGNN()
        self.qlora = QLoRALayer(128)
        self.llm = LLMModule(128)

    def forward(self, x):
        x = self.gnn(x)
        x = self.qlora(x)
        return self.llm(x)

def reward_fn(pred, label):
    pred = torch.argmax(pred, dim=1)
    return (pred == label).float().mean()

def train():
    data = preprocess(load_dataset())
    vocab = build_vocab(data)

    kf = KFold(n_splits=10, shuffle=True)
    all_preds, all_labels = [], []

    for train_idx, test_idx in kf.split(data):
        model = FullModel().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)
        loss_fn = nn.CrossEntropyLoss()

        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]

        for epoch in range(100):
            model.train()
            random.shuffle(train_data)

            for i in range(0, len(train_data), 64):
                batch = train_data[i:i+64]

                inputs, labels = [], []
                for text, _, label in batch:
                    inputs.append(encode_text(text, vocab))
                    labels.append(label)

                inputs = torch.tensor(inputs).to(DEVICE)
                labels = torch.tensor(labels).to(DEVICE)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                reward = reward_fn(outputs, labels)
                loss = loss - 0.1 * reward

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            for text, _, label in test_data:
                inp = torch.tensor([encode_text(text, vocab)]).to(DEVICE)
                pred = torch.argmax(model(inp), dim=1).item()
                all_preds.append(pred)
                all_labels.append(label)

    evaluate(all_preds, all_labels)

if __name__ == "__main__":
    train()
	