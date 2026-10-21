"""
Dog breed classification (ResNet50 fine-tuning) - SageMaker SDK v3 version
(no SageMaker Debugger / smdebug, same approach as pytorch_cifar_profiling.py).

What replaces what:
  - DebuggerHookConfig (losses, gradients)  -> TensorBoard SummaryWriter + metrics.csv
  - ProfilerConfig (system monitor)         -> background thread sampling CPU/GPU/memory
                                               -> TensorBoard + system_metrics.csv
  - Framework profiling                     -> torch.profiler (trace + summary table)
  - Debugger rules                          -> same checks done in-script -> rules.json

On SageMaker, everything is written to /opt/ml/output/data/, which is uploaded automatically
to S3 as output.tar.gz (next to model.tar.gz). TensorBoard logs also go to
/opt/ml/output/tensorboard (uploaded live if TensorBoardOutputConfig is set).
When run locally (notebook debugging block), outputs go to ./output/ instead.
"""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# In a training container /opt/ml/input/config always exists. Locally (SageMaker Studio
# notebook) /opt/ml is not writable, so fall back to ./output for the debugging block.
ON_SAGEMAKER = "SM_MODEL_DIR" in os.environ or os.path.isdir("/opt/ml/input/config")
OUTPUT_DIR = "/opt/ml/output/data" if ON_SAGEMAKER else "output/data"          # -> output.tar.gz
TB_DIR = "/opt/ml/output/tensorboard" if ON_SAGEMAKER else "output/tensorboard"  # -> TensorBoardOutputConfig


# ----------------------------------------------------------------------------------
# "Profiler" equivalent #1: system monitor (CPU / GPU / memory), like ProfilerConfig
# ----------------------------------------------------------------------------------
class SystemMonitor(threading.Thread):
    def __init__(self, writer, interval_s=0.5):
        super().__init__(daemon=True)
        self.writer = writer
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.rows = []
        self.t_start = time.time()

    @staticmethod
    def _gpu_stats():
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                text=True,
            ).strip().splitlines()[0]
            util, used, total = [float(x) for x in out.split(",")]
            return util, 100.0 * used / total
        except Exception:
            return None, None

    def run(self):
        psutil.cpu_percent(interval=None)  # first call initialises the counter
        step = 0
        while not self.stop_event.is_set():
            time.sleep(self.interval_s)
            t = time.time() - self.t_start
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            gpu, gpu_mem = self._gpu_stats()
            self.rows.append({"time_s": round(t, 2), "cpu_pct": cpu, "mem_pct": mem,
                              "gpu_pct": gpu, "gpu_mem_pct": gpu_mem})
            self.writer.add_scalar("system/cpu_pct", cpu, step)
            self.writer.add_scalar("system/mem_pct", mem, step)
            if gpu is not None:
                self.writer.add_scalar("system/gpu_pct", gpu, step)
                self.writer.add_scalar("system/gpu_mem_pct", gpu_mem, step)
            step += 1

    def stop(self):
        self.stop_event.set()
        self.join()
        with open(f"{OUTPUT_DIR}/system_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["time_s", "cpu_pct", "mem_pct", "gpu_pct", "gpu_mem_pct"])
            w.writeheader()
            w.writerows(self.rows)


# ----------------------------------------------------------------------------------
# "Debugger" equivalent: helpers
# ----------------------------------------------------------------------------------
class RunLog:
    """Everything the old Debugger hook used to collect, shared by train() and evaluate()."""
    def __init__(self, writer):
        self.writer = writer
        self.metrics_rows = []    # step-level losses (TRAIN / EVAL), like the "losses" collection
        self.train_step = 0
        self.eval_step = 0
        self.current_epoch = 0
        self.timing = {"data_loading_s": 0.0, "compute_s": 0.0}


def mean_abs_grad(model):
    # Same quantity the vanishing_gradient rule looks at.
    # The backbone is frozen, so only the new fc head has gradients.
    grads = [p.grad.abs().mean() for p in model.parameters() if p.grad is not None]
    return torch.stack(grads).mean().item() if grads else 0.0


def evaluate(model, loader, criterion, device, name, log=None, mode=None):
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            batch_loss = criterion(out, y).item()
            loss += batch_loss * x.size(0)
            correct += (out.argmax(1) == y).sum().item()  # example : tensor(27).item() = 27
            if log is not None and mode is not None:
                log.writer.add_scalar(f"{mode}/loss", batch_loss, log.eval_step)
                log.metrics_rows.append({"mode": mode, "epoch": log.current_epoch, "step": log.eval_step,
                                         "loss": batch_loss, "mean_abs_grad": ""})
                log.eval_step += 1
    n = len(loader.dataset)
    logger.info(f"{name} set: Average loss: {loss/n:.4f}, Accuracy: {100*correct/n:.2f}%")
    return loss / n, correct / n


def train(model, loader, criterion, optimizer, device, epoch, log, profile_steps=False):
    model.train()
    running = 0.0

    # "Profiler" equivalent #2: framework profiling on a few steps (first epoch only)
    prof = None
    if profile_steps:
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
        prof = profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=tensorboard_trace_handler(f"{TB_DIR}/profiler"),
            record_shapes=True,
            profile_memory=True,
        )
        prof.start()

    t0 = time.time()
    for x, y in loader:
        t1 = time.time()
        log.timing["data_loading_s"] += t1 - t0

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        g = mean_abs_grad(model)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()   # accurate compute timing
        t0 = time.time()
        log.timing["compute_s"] += t0 - t1

        # Step-level logging (save_interval = 1: every step)
        batch_loss = loss.item()
        log.writer.add_scalar("TRAIN/loss", batch_loss, log.train_step)
        log.writer.add_scalar("TRAIN/mean_abs_grad", g, log.train_step)
        log.metrics_rows.append({"mode": "TRAIN", "epoch": epoch, "step": log.train_step,
                                 "loss": batch_loss, "mean_abs_grad": g})
        log.train_step += 1
        running += batch_loss * x.size(0)  # This converts the average batch loss into the batch's contribution to the accumulated loss.
        if prof is not None:
            prof.step()

    if prof is not None:
        prof.stop()
        sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
        try:
            table = prof.key_averages().table(sort_by=sort_key, row_limit=15)
        except Exception:  # newer torch versions renamed cuda_time_total -> device_time_total
            table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=15)
        print(table)
        with open(f"{OUTPUT_DIR}/torch_profiler_summary.txt", "w") as f:
            f.write(table)

    # Weight / gradient histograms once per epoch (replaces the "weights"/"gradients" collections).
    # Only the trainable head: the frozen backbone weights never change.
    for name, p in model.named_parameters():
        if p.requires_grad:
            log.writer.add_histogram(f"weights/{name}", p, epoch)
            if p.grad is not None:
                log.writer.add_histogram(f"gradients/{name}", p.grad, epoch)

    avg = running / len(loader.dataset)
    logger.info(f"Epoch {epoch}: Train loss: {avg:.4f}")
    return avg


def net(args):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for p in model.parameters():          # freeze the pretrained backbone
        p.requires_grad = False
    model.fc = nn.Sequential(             # new trainable head
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, args.num_classes),
    )
    return model


def create_data_loaders(args):  # (data, batch_size):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm
    ])

    train = datasets.ImageFolder(args.train_dir, train_tf)
    valid = datasets.ImageFolder(args.valid_dir, eval_tf)
    test = datasets.ImageFolder(args.test_dir, eval_tf)

    return mk(train, True, args), mk(valid, False, args), mk(test, False, args)


def mk(ds, sh, args):
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                       shuffle=sh,
                                       num_workers=4)


# ----------------------------------------------------------------------------------
# "Rules" equivalent: same checks as the built-in Debugger rules, evaluated in-script
# ----------------------------------------------------------------------------------
def evaluate_rules(metrics_rows, epoch_summary, timing, sys_rows, test_result):
    train_losses = [r["loss"] for r in metrics_rows if r["mode"] == "TRAIN"]
    grads = [r["mean_abs_grad"] for r in metrics_rows if r["mode"] == "TRAIN"]
    results = {}

    # LossNotDecreasing: compare mean loss of last 5 steps vs first 5 steps
    w = min(5, len(train_losses) // 2) or 1
    first, last = np.mean(train_losses[:w]), np.mean(train_losses[-w:])
    results["LossNotDecreasing"] = {"triggered": bool(last >= first * 0.9),
                                    "detail": f"first {w} steps={first:.3f}, last {w} steps={last:.3f}"}

    # VanishingGradient: mean |grad| below 1e-7
    min_g = float(np.min(grads)) if grads else 0.0
    results["VanishingGradient"] = {"triggered": bool(min_g < 1e-7), "detail": f"min mean|grad|={min_g:.2e}"}

    # Overfit: validation loss much higher than training loss
    last_ep = epoch_summary[-1]
    ratio = last_ep["val_loss"] / max(last_ep["train_loss"], 1e-12)
    results["Overfit"] = {"triggered": bool(ratio > 1.5), "detail": f"val/train loss ratio={ratio:.2f}"}

    # Overtraining: validation loss stopped improving
    val = [e["val_loss"] for e in epoch_summary]
    worse = len(val) > 1 and val[-1] >= min(val[:-1])
    results["Overtraining"] = {"triggered": bool(worse), "detail": f"val losses per epoch={[round(v, 3) for v in val]}"}

    # ProfilerReport (light version): data-loading bottleneck + low GPU utilisation
    total = timing["data_loading_s"] + timing["compute_s"]
    data_share = timing["data_loading_s"] / total if total else 0.0
    results["DataLoaderBottleneck"] = {"triggered": bool(data_share > 0.3),
                                       "detail": f"{100 * data_share:.0f}% of step time spent waiting for data"}
    gpu = [r["gpu_pct"] for r in sys_rows if r["gpu_pct"] is not None]
    if gpu:
        p50 = float(np.percentile(gpu, 50))
        results["LowGPUUtilization"] = {"triggered": bool(p50 < 70), "detail": f"median GPU utilisation={p50:.0f}%"}

    for name, r in results.items():
        logger.info(f"[RULE] {name}: {'TRIGGERED' if r['triggered'] else 'ok'} ({r['detail']})")
    with open(f"{OUTPUT_DIR}/rules.json", "w") as f:
        json.dump({"rules": results, "timing": timing, "epochs": epoch_summary, "test": test_result},
                  f, indent=2)


def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    # Check if the instance has a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize a model by calling the net function
    model = net(args).to(device)

    # Create your loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)  # we are only training the custom fully connected layers we added at the end of the model architecture
    train_loader, valid_loader, test_loader = create_data_loaders(args)

    # Debugger/profiler equivalents
    log = RunLog(SummaryWriter(log_dir=TB_DIR))
    monitor = SystemMonitor(log.writer, interval_s=0.5)   # same 500 ms as the old ProfilerConfig
    monitor.start()
    epoch_summary = []

    # Call the train function to start training your model
    for epoch in range(1, args.epochs+1):
        log.current_epoch = epoch
        start = time.time()
        train_loss = train(model, train_loader, loss_criterion, optimizer, device, epoch, log,
                           profile_steps=(epoch == 1))
        val_loss, val_acc = evaluate(model, valid_loader, loss_criterion, device, "Validation",
                                     log=log, mode="EVAL")
        log.writer.add_scalar("epoch/train_loss", train_loss, epoch)
        log.writer.add_scalar("epoch/val_loss", val_loss, epoch)
        log.writer.add_scalar("epoch/val_accuracy", val_acc, epoch)
        epoch_summary.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                              "val_acc": val_acc, "time_s": time.time() - start})

    # Test the model to see its accuracy
    # After all epochs are finished, we can reuse the same evaluate function with the test dataset
    test_loss, test_acc = evaluate(model, test_loader, loss_criterion, device, "Test")

    monitor.stop()
    with open(f"{OUTPUT_DIR}/metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "epoch", "step", "loss", "mean_abs_grad"])
        w.writeheader()
        w.writerows(log.metrics_rows)
    evaluate_rules(log.metrics_rows, epoch_summary, log.timing, monitor.rows,
                   {"test_loss": test_loss, "test_acc": test_acc})
    log.writer.flush()
    log.writer.close()

    # Copy TensorBoard logs into output.tar.gz too, so they're always retrievable
    shutil.copytree(TB_DIR, f"{OUTPUT_DIR}/tensorboard", dirs_exist_ok=True)

    # Save the trained model
    # Anything written to SM_MODEL_DIR is packaged into model.tar.gz and uploaded to S3.
    # Move to CPU before saving so the weights load on any machine, GPU or not.
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(),  # get model learned weights and biases
               os.path.join(args.model_dir, "model.pth"))  # creates the save path
    # Save the breed names (ImageFolder's alphabetical index -> folder name order)
    # so inference.py can return breed names instead of class indices
    with open(os.path.join(args.model_dir, "classes.json"), "w") as f:
        json.dump(train_loader.dataset.classes, f)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # Both spellings accepted, so hyperparameter keys like "batch-size" or "batch_size" both work
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--num-classes", "--num_classes", dest="num_classes", type=int, default=133)
    p.add_argument("--train-dir", default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--valid-dir", default=os.environ.get("SM_CHANNEL_VALID"))
    p.add_argument("--test-dir",  default=os.environ.get("SM_CHANNEL_TEST"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    args = p.parse_args()
    for key, value in vars(args).items():
        print(f"{key}:{value}")
    main(args)
