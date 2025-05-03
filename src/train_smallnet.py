import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm


# ----- 1. Arguments -----
parser = argparse.ArgumentParser()
parser.add_argument(
    '--loss', choices=['crossentropy', 'kldiv'], default='crossentropy',
    help="Choose 'crossentropy' or 'kldiv'"
)
parser.add_argument(
    '--use_soft', action='store_true',
    help="If set, train with soft labels"
)
args = parser.parse_args()


# ----- 2. Setup -----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
batch_size = 128

train_ds = torchvision.datasets.CIFAR10(
    root='cifar10/', train=True, download=True, transform=transform
)
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=not args.use_soft,
    sampler=SequentialSampler(train_ds) if args.use_soft else None
)

test_ds = torchvision.datasets.CIFAR10(
    root='cifar10/', train=False, download=True, transform=transform
)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# ----- 3. Models -----
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3,	padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----- 4. Train function -----
def train():
    set_seed()
    model = SmallNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 25

    if args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.KLDivLoss(reduction='batchmean')

    if args.use_soft:
        soft_labels = torch.load('soft_labels.pt')

    history = {'losses': [], 'accuracies': []}

    print(f"\nTraining SmallNet (loss={args.loss}, use_soft={args.use_soft})")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            if args.use_soft:
                targets = soft_labels[idx*batch_size:(idx+1)*batch_size].to(device)
                log_probs = F.log_softmax(model(x), dim=1)
                loss = criterion(log_probs, targets)
            else:
                logits = model(x)
                if args.loss == 'crossentropy':
                    loss = criterion(logits, y)
                else:
                    log_probs = F.log_softmax(logits, dim=1)
                    one_hot = F.one_hot(y, num_classes=10).float().to(device)
                    loss = criterion(log_probs, one_hot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        avg_loss = running_loss / total
        acc = correct / total
        history['losses'].append(avg_loss)
        history['accuracies'].append(acc)

        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

    elapsed = (time.time() - start_time) / 60
    torch.save((model.state_dict(), history), f"smallnet_{args.loss}_{args.use_soft}.pth")
    print(f"\nFinished in {elapsed:.2f} min")


if __name__ == '__main__':
    train()
