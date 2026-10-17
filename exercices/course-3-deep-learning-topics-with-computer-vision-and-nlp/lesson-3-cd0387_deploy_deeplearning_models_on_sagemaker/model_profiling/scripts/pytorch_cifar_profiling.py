"""
CIFAR-10 training script - SageMaker SDK v3 version (no SageMaker Debugger / smdebug).

What replaces what:
  - DebuggerHookConfig (losses, gradients)  -> TensorBoard SummaryWriter + metrics.csv
  - ProfilerConfig (system monitor)         -> background thread sampling CPU/GPU/memory
                                               -> TensorBoard + system_metrics.csv
  - Framework profiling                     -> torch.profiler (trace + summary table)
  - Debugger rules                          -> same checks done in-script -> rules.json

Everything is written to /opt/ml/output/data/, which SageMaker uploads automatically
to S3 as output.tar.gz (next to model.tar.gz). TensorBoard logs also go to
/opt/ml/output/tensorboard (uploaded live if TensorBoardOutputConfig is set).
"""

import os


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import csv
import json
import shutil
import subprocess
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


OUTPUT_DIR = "/opt/ml/output/data"          # -> s3://.../<job>/output/output.tar.gz
TB_DIR = "/opt/ml/output/tensorboard"       # -> TensorBoardOutputConfig S3 path (if set)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)


def str2bool(v):
    # Replaces smdebug.profiler.utils.str2bool
    return str(v).lower() in ("true", "1", "yes", "y")


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
def mean_abs_grad(net):
    # Same quantity the vanishing_gradient rule looks at
    grads = [p.grad.abs().mean() for p in net.parameters() if p.grad is not None]
    return torch.stack(grads).mean().item() if grads else 0.0


def train(args, net, device, writer):
    batch_size = args.batch_size
    epoch = args.epoch

    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    # transform_valid = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,  download=False, transform=transform)
    validset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)

    loss_optim = nn.CrossEntropyLoss()
    # NB: lr=1.0 is (deliberately) very high in the original exercise -> the rules should fire
    optimizer = optim.SGD(net.parameters(), lr=1.0, momentum=0.9)

    metrics_rows = []         # step-level losses (TRAIN / EVAL), like the "losses" collection
    epoch_summary = []        # epoch-level averages, used by the rules
    epoch_times = []
    train_step, eval_step = 0, 0
    timing = {"data_loading_s": 0.0, "compute_s": 0.0}

    for i in range(epoch):
        print("START TRAINING")
        start = time.time()
        net.train()
        train_loss, n_train_batches = 0.0, 0

        # "Profiler" equivalent #2: framework profiling on a few steps of the first epoch
        prof = None
        if i == 0:
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
        for inputs, targets in trainloader:
            t1 = time.time()
            timing["data_loading_s"] += t1 - t0

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_optim(outputs, targets)
            loss.backward()
            g = mean_abs_grad(net)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()   # accurate compute timing
            t0 = time.time()
            timing["compute_s"] += t0 - t1

            # Step-level logging (save_interval = 1: every step)
            writer.add_scalar("TRAIN/loss", loss.item(), train_step)
            writer.add_scalar("TRAIN/mean_abs_grad", g, train_step)
            metrics_rows.append({"mode": "TRAIN", "epoch": i, "step": train_step,
                                 "loss": loss.item(), "mean_abs_grad": g})
            train_step += 1
            train_loss += loss.item()
            n_train_batches += 1
            if prof is not None:
                prof.step()

        if prof is not None:
            prof.stop()
            table = prof.key_averages().table(
                sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=15)
            print(table)
            with open(f"{OUTPUT_DIR}/torch_profiler_summary.txt", "w") as f:
                f.write(table)

        # Weight / gradient histograms once per epoch (replaces the "weights"/"gradients" collections)
        for name, p in net.named_parameters():
            writer.add_histogram(f"weights/{name}", p, i)
            if p.grad is not None:
                writer.add_histogram(f"gradients/{name}", p.grad, i)

        print("START VALIDATING")
        net.eval()
        val_loss, n_val_batches, correct, total = 0.0, 0, 0, 0
        with torch.no_grad():
            for inputs, targets in validloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = loss_optim(outputs, targets)
                writer.add_scalar("EVAL/loss", loss.item(), eval_step)
                metrics_rows.append({"mode": "EVAL", "epoch": i, "step": eval_step,
                                     "loss": loss.item(), "mean_abs_grad": ""})
                eval_step += 1
                val_loss += loss.item()
                n_val_batches += 1
                correct += (outputs.argmax(1) == targets).sum().item()
                total += targets.size(0)

        # Average per batch (the original summed, which makes train vs val incomparable)
        avg_train, avg_val, val_acc = train_loss / n_train_batches, val_loss / n_val_batches, correct / total
        writer.add_scalar("epoch/train_loss", avg_train, i)
        writer.add_scalar("epoch/val_loss", avg_val, i)
        writer.add_scalar("epoch/val_accuracy", val_acc, i)
        epoch_summary.append({"epoch": i, "train_loss": avg_train, "val_loss": avg_val, "val_acc": val_acc})

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        print("Epoch %d: train loss %.3f, val loss %.3f, val acc %.3f, in %.1f sec"
              % (i, avg_train, avg_val, val_acc, epoch_time))

    with open(f"{OUTPUT_DIR}/metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "epoch", "step", "loss", "mean_abs_grad"])
        w.writeheader()
        w.writerows(metrics_rows)

    return float(np.percentile(epoch_times, 50)), metrics_rows, epoch_summary, timing


# ----------------------------------------------------------------------------------
# "Rules" equivalent: same checks as the built-in Debugger rules, evaluated in-script
# ----------------------------------------------------------------------------------
def evaluate_rules(metrics_rows, epoch_summary, timing, sys_rows):
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
        print(f"[RULE] {name}: {'TRIGGERED' if r['triggered'] else 'ok'} ({r['detail']})")
    with open(f"{OUTPUT_DIR}/rules.json", "w") as f:
        json.dump({"rules": results, "timing": timing, "epochs": epoch_summary}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/tmp/data"))  # Added
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))  # Added

    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}:{value}")

    model = models.__dict__[args.model](weights="DEFAULT")
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    writer = SummaryWriter(log_dir=TB_DIR)
    monitor = SystemMonitor(writer, interval_s=0.5)   # same 500 ms as the old ProfilerConfig
    monitor.start()

    median_time, metrics_rows, epoch_summary, timing = train(args, model, device, writer)

    monitor.stop()
    evaluate_rules(metrics_rows, epoch_summary, timing, monitor.rows)
    writer.flush()
    writer.close()

    # Copy TensorBoard logs into output.tar.gz too, so they're always retrievable
    shutil.copytree(TB_DIR, f"{OUTPUT_DIR}/tensorboard", dirs_exist_ok=True)
    print("Median training time per Epoch=%.1f sec" % median_time)

    # Added
    # Anything written to SM_MODEL_DIR is packaged into model.tar.gz and uploaded to S3.
    # Move to CPU before saving so the weights load on any machine, GPU or not.
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == "__main__":
    main()
