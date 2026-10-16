import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 3 x 32 x 32 (CIFAR10 RGB images)
        self.conv1 = nn.Conv2d(3, 6, 5)        # -> 6 x 28 x 28
        self.pool = nn.MaxPool2d(2, 2)         # halves spatial dims
        self.conv2 = nn.Conv2d(6, 16, 5)       # -> 16 x 10 x 10 (after pool: 16 x 5 x 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)           # 10 CIFAR10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # log_softmax because train/test use F.nll_loss
        return F.log_softmax(x, dim=1)


def train(model, train_loader, optimizer, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move each batch to the same device as the model
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Example")
    # SageMaker passes each hyperparameter as a CLI argument, e.g. --epochs 2
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    # SageMaker-provided paths (fall back to local dirs when run outside SageMaker)
    # parser.add_argument("--data-dir", type=str, default="/tmp/data")
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/tmp/data"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    args = parser.parse_args()

    # Use the GPU if the instance has one, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Datasets loaded directly from S3 bucket
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform)

    # pin_memory speeds up host-to-GPU copies; it's a no-op benefit on CPU
    loader_kwargs = {"num_workers": 2, "pin_memory": device.type == "cuda"}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs
    )

    model = Net().to(device)

    # Create the optimizer *after* moving the model, so it references the device parameters
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch, device)
        test(model, test_loader, device)

    # Anything written to SM_MODEL_DIR is packaged into model.tar.gz and uploaded to S3.
    # Move to CPU before saving so the weights load on any machine, GPU or not.
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == "__main__":
    main()
