import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ----- 1. Setup -----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
test_ds = torchvision.datasets.CIFAR10(
    root='cifar10/', train=False, download=True, transform=transform
)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

# ----- 2. Load & Evaluate -----
results = {}
for loss in ['crossentropy', 'kldiv']:
    for use_soft in [False, True]:
        path = f"smallnet_{loss}_{use_soft}.pth"
        try:
            state, history = torch.load(path)
        except FileNotFoundError:
            continue

        # rebuild model
        from train_smallnet import SmallNet
        model = SmallNet().to(device)
        model.load_state_dict(state)
        model.eval()

        # test accuracy
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)
        acc = correct / total
        results[(loss, use_soft)] = (history['losses'], history['accuracies'], acc)

# ----- 3. Plot -----
plt.figure(figsize=(8, 5))
for (loss, use_soft), (losses, _, _) in results.items():
    label = f"{loss} soft={use_soft}"
    plt.plot(losses, label=label)
plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

print("\nFinal Test Accuracies:")
for (loss, use_soft), (_, _, acc) in results.items():
    print(f"{loss} (soft={use_soft}): {acc:.4f}")
