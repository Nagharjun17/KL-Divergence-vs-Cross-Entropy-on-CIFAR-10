import torch
import torch.nn.functional as F
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SequentialSampler

from train_resnet50 import ResNet50  # assumes train_resnet50.py is in the same folder


# ----- 1. Setup -----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
batch_size = 128

dataset = torchvision.datasets.CIFAR10(
    root='cifar10/',
    train=True,
    download=True,
    transform=transform
)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    sampler=SequentialSampler(dataset)
)


# ----- 2. Load Teacher -----
model = ResNet50().to(device)
model.load_state_dict(torch.load('resnet50_cifar10.pth'))
model.eval()


# ----- 3. Generate Soft Labels -----
soft_labels = []
true_labels = []

with torch.no_grad():
    for x, y in tqdm(loader, desc="Generating soft labels"):
        x = x.to(device)
        probs = F.softmax(model(x), dim=1)
        soft_labels.append(probs.cpu())
        true_labels.append(y)

soft_labels = torch.cat(soft_labels, dim=0)
true_labels = torch.cat(true_labels, dim=0)

torch.save(soft_labels, 'soft_labels.pt')
torch.save(true_labels, 'true_labels.pt')
print("Saved soft_labels.pt and true_labels.pt")
