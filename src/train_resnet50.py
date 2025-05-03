import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ----- 1. Setup -----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
batch_size = 128

traindata = torchvision.datasets.CIFAR10(
    root='cifar10/',
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True)


# ----- 2. ResNet50 Teacher Model -----
class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(weights='DEFAULT')
        self.final = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.resnet(x)
        return self.final(x)


# ----- 3. Train ResNet50 -----
def train():
    model = ResNet50().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100

    print("\nTraining ResNet50...")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss=running_loss/total, acc=correct/total)

    elapsed = (time.time() - start_time) / 60
    print(f"\nResNet50 Training Time: {elapsed:.2f} minutes")
    torch.save(model.state_dict(), 'resnet50_cifar10.pth')


if __name__ == '__main__':
    train()
