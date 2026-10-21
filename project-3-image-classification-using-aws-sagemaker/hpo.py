"""
Dog breed classification (ResNet50 fine-tuning) - HPO entry point.

This is intentionally a stripped-down version of train_model.py:
  - No TensorBoard, no system monitor, no torch.profiler, no rules.
  - Tunes on validation loss only (no test set — avoids data leakage).
  - Prints the metric in the exact format SageMaker's HPO tuner expects
    so it can parse and compare trials automatically.
  - Saves no model artifact: HPO trials are throwaway runs.

Hyperparameters tuned:
  --lr          Learning rate for Adam (log-uniform range, e.g. 1e-4 to 1e-1).
                The most impactful HP for fine-tuning a frozen backbone.
  --batch-size  Mini-batch size (categorical: 32, 64, 128).
                Affects convergence speed and GPU memory usage.
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def evaluate(model, loader, criterion, device, name):
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    avg_loss = loss / n
    accuracy = 100.0 * correct / n
    logger.info(f"{name} set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def train(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    avg = running / len(loader.dataset)
    logger.info(f"Epoch {epoch}: Train loss: {avg:.4f}")
    return avg


def net(args):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for p in model.parameters():          # freeze the pretrained backbone
        p.requires_grad = False
    model.fc = nn.Sequential(             # new trainable head (same architecture as train_model.py)
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, args.num_classes),
    )
    return model


def create_data_loaders(args):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm,
    ])

    train_ds = datasets.ImageFolder(args.train_dir, train_tf)
    valid_ds = datasets.ImageFolder(args.valid_dir, eval_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = net(args).to(device)

    loss_criterion = nn.CrossEntropyLoss()
    # Only the fc head is trainable; passing model.fc.parameters() keeps
    # the frozen backbone out of the optimizer entirely.
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    train_loader, valid_loader = create_data_loaders(args)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, loss_criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, valid_loader, loss_criterion, device, "Validation")
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # SageMaker's HPO tuner scrapes stdout for a line matching the regex defined
    # in metric_definitions. The format below is what the tuner will look for:
    #   {"Name": "validation:loss", "Regex": "Validation Loss: ([0-9\\.]+)"}
    # Using the best (lowest) validation loss across all epochs is more reliable
    # than the final epoch's loss, especially when the number of epochs is small.
    # Plain print (not logger) keeps stdout clean for SageMaker's HPO regex scraper.
    # A logger line would add an "INFO:__main__:" prefix that complicates the regex.
    print(f"Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Hyperparameters to tune
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=32)
    # Fixed hyperparameters (not tuned, passed as static_hyperparameters in the tuner)
    p.add_argument("--epochs", type=int, default=2)       # keep trials short
    p.add_argument("--num-classes", type=int, default=133)
    p.add_argument("--train-dir", default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--valid-dir", default=os.environ.get("SM_CHANNEL_VALID"))
    # SageMaker model output directory, defaults to "model" when running locally
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    args = p.parse_args()
    for key, value in vars(args).items():
        print(f"{key}:{value}")
    main(args)
