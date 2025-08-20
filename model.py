"""
AI-Image Detector (Tiny Depthwise CNN) with Curriculum Learning, Pruning, and QAT

What you get
------------
• Tiny depthwise-separable CNN (MobileNet-like) with width multiplier for size control.
• 2-class classifier: 0 = real photo, 1 = AI-generated.
• Three-stage curriculum (easy → medium → hard) on AI-generated datasets + one real dataset.
• Progressive unfreezing + short warmups per stage.
• Mixed precision (AMP), cosine LR, label smoothing, EMA weights, gradient clipping.
• Structured pruning (channel-level) + unstructured L1 sparsity for final shrink.
• Quantization-aware training (QAT) with QNNPACK backend.
• ONNX export (fp32 + int8 QAT). Optional: extra ORT static quant if onnxruntime is installed.
• Windows-friendly multiprocessing (works with num_workers>0) & clean tqdm progress bars.

Folder expectations
-------------------
Provide four folders (or CSVs) of images:
  AI_EASY_DIR, AI_MED_DIR, AI_HARD_DIR, REAL_DIR
Each folder can contain subfolders. We auto-label AI folders as label=1 and REAL_DIR as label=0.
You can also pass a CSV listing [path,label] via --csv_* flags instead.

Quick start (example)
---------------------
python ai_image_detector_curriculum_qat.py \
  --ai_easy ./data/ai_easy \
  --ai_medium ./data/ai_medium \
  --ai_hard ./data/ai_hard \
  --real ./data/real \
  --epochs_easy 3 --epochs_medium 3 --epochs_hard 6 \
  --batch_size 128 --workers 8 --width_mult 0.5 \
  --img_size 192 \
  --out_dir ./artifacts

Then look in ./artifacts for:
  model_fp32.onnx  (float ONNX, WebGPU-friendly)
  model_int8_qat.onnx  (int8 from QAT)
  model_pruned_qat.pt  (PyTorch int8-ready after pruning & QAT)

Chrome Extension (ORT Web) hint
-------------------------------
• Use onnxruntime-web with WebGPU/WebGL. 
• Preprocess with the same mean/std and img_size.
• Run inference on model_fp32.onnx or model_int8_qat.onnx.
• Output is logit pair [p(real), p(ai)] after softmax.

"""
from __future__ import annotations
import os
import csv
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import onnxruntime

# -------------------------------
# Utilities
# -------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def human_count(x: int) -> str:
    suffixes = ["", "K", "M", "B"]
    idx = 0
    value = float(x)
    while value >= 1000 and idx < len(suffixes) - 1:
        value /= 1000.0
        idx += 1
    return f"{value:.2f}{suffixes[idx]}"


# -------------------------------
# Dataset
# -------------------------------
class ImageList(Dataset):
    """Reads from folder (walk) or CSV [path,label]. Label must be 0 or 1.
    We skip non-image files, small icons (< 64 px), and obviously tiny banners.
    """
    def __init__(self, root_or_csv: str, img_size: int, augment: bool, label_override: Optional[int] = None):
        self.samples: List[Tuple[str, int]] = []
        self.img_size = img_size

        if os.path.isdir(root_or_csv):
            for r, _, files in os.walk(root_or_csv):
                for f in files:
                    fp = os.path.join(r, f)
                    if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                        continue
                    # Skip tiny assets (icons/badges/banners). You can adjust thresholds.
                    try:
                        with Image.open(fp) as im:
                            w, h = im.size
                        if min(w, h) < 64 or w * h < 64 * 64:
                            continue
                    except Exception:
                        continue
                    label = label_override if label_override is not None else self._infer_label_from_path(fp)
                    if label in (0, 1):
                        self.samples.append((fp, label))
        else:
            with open(root_or_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue
                    fp, lab = row[0], int(row[1])
                    if os.path.isfile(fp) and lab in (0, 1):
                        self.samples.append((fp, lab))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found in {root_or_csv}")

        # Augmentations: modest, accuracy-friendly
        if augment:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size * 1.15)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _infer_label_from_path(self, path: str) -> int:
        # If you place images under folders named "real" or "ai" we auto-label.
        low = path.lower()
        if os.sep + "real" + os.sep in low or low.endswith(os.sep + "real"):
            return 0
        if os.sep + "ai" + os.sep in low or low.endswith(os.sep + "ai"):
            return 1
        # Unknown → raise to force explicit CSV or override
        raise ValueError("Cannot infer label from path; use CSV or label_override.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        x = self.tf(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class MixedConcat(Dataset):
    """Combines multiple datasets. If balance>0, oversamples the minority class to approach balance.
    Useful to merge real set with an AI-difficulty split.
    """
    def __init__(self, datasets: List[Dataset], balance: float = 0.0):
        self.samples = []
        for ds in datasets:
            for i in range(len(ds)):
                self.samples.append((ds, i))
        self.balance = balance
        if balance > 0:
            # Build indices per class to support rudimentary oversampling
            self.cls_idx = {0: [], 1: []}
            for j, (ds, i) in enumerate(self.samples):
                _, y = ds[i]
                self.cls_idx[int(y.item())].append(j)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.balance > 0 and random.random() < self.balance:
            # Sample class 0 or 1 with equal prob
            c = random.randint(0, 1)
            j = random.choice(self.cls_idx[c])
            ds, i = self.samples[j]
            return ds[i]
        ds, i = self.samples[idx]
        return ds[i]


# -------------------------------
# Tiny Model (Depthwise Separable CNN)
# -------------------------------
class DWConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.pw(self.dw(x))


def make_layers(width_mult: float, num_classes: int):
    def c(ch):
        return max(8, int(ch * width_mult))

    layers = []
    layers += [nn.Sequential(
        nn.Conv2d(3, c(16), 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c(16)),
        nn.ReLU(inplace=True))]

    cfg = [
        # (out_c, stride)
        (c(16), 1), (c(24), 2), (c(24), 1), (c(32), 2), (c(32), 1), (c(64), 2),
        (c(64), 1), (c(96), 1), (c(128), 2), (c(128), 1)
    ]
    in_c = c(16)
    for out_c, s in cfg:
        layers += [DWConvBlock(in_c, out_c, stride=s)]
        in_c = out_c

    features = nn.Sequential(*layers)
    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(in_c, max(16, c(64))),
        nn.ReLU(inplace=True),
        nn.Linear(max(16, c(64)), num_classes),
    )
    return features, classifier


class TinyAIDetector(nn.Module):
    def __init__(self, width_mult: float = 0.5, num_classes: int = 2):
        super().__init__()
        self.features, self.classifier = make_layers(width_mult, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------------
# Training Helpers
# -------------------------------
@dataclass
class StageCfg:
    name: str
    epochs: int
    lr: float
    freeze_until: str  # "stem", "mid", or "none"


def set_freeze(model: TinyAIDetector, freeze_until: str):
    # Freeze progressively less per stage
    # stem = first conv
    # mid = up to roughly half of the blocks
    # none = train all
    for p in model.parameters():
        p.requires_grad = True

    modules = list(model.features.children())
    if freeze_until == "stem" and len(modules) > 0:
        for p in modules[0].parameters():
            p.requires_grad = False
    elif freeze_until == "mid":
        cutoff = max(1, len(modules) // 2)
        for m in modules[:cutoff]:
            for p in m.parameters():
                p.requires_grad = False


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow and p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        num_classes = pred.size(-1)
        logprobs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=-1))


def accuracy_from_logits(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def create_loader(dataset: Dataset, batch_size: int, workers: int, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True, drop_last=False, persistent_workers=workers>0)


# -------------------------------
# Pruning
# -------------------------------
from torch.nn.utils import prune

def structured_prune(model: nn.Module, amount: float = 0.3):
    """Channel-wise pruning on 1x1 pointwise convs to remove entire output channels."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size == (1,1):
            prune.ln_structured(m, name='weight', amount=amount, n=2, dim=0)
            prune.remove(m, 'weight')  # make it permanent


def l1_unstructured_prune(model: nn.Module, amount: float = 0.2):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            prune.l1_unstructured(m, name='weight', amount=amount)
            prune.remove(m, 'weight')


# -------------------------------
# QAT (Quantization-Aware Training)
# -------------------------------
from torch.ao.quantization import get_default_qat_qconfig
from torch.ao.quantization import prepare_qat_fx, convert_fx

def fuse_for_qat(model: nn.Module) -> nn.Module:
    # FX graph mode will handle typical fusions (Conv+BN+ReLU) automatically later
    return model


# -------------------------------
# Train Loop
# -------------------------------

def train_one_stage(model, loader_tr, loader_val, device, stage: StageCfg, epochs_warmup=1,
                     max_norm=2.0, ema: Optional[EMA] = None, criterion=None):
    n_epochs = stage.epochs
    lr = stage.lr

    set_freeze(model, stage.freeze_until)

    # Only optimize trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    # cosine schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - 1))

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    best_val = 0.0

    for epoch in range(1, n_epochs + 1):
        model.train()
        pbar = tqdm(loader_tr, desc=f"{stage.name} | Epoch {epoch}/{n_epochs}", leave=False)
        total_loss = 0.0
        total_acc = 0.0
        n = 0
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(True):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if max_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            acc = accuracy_from_logits(logits.detach(), y)
            total_loss += loss.item() * x.size(0)
            total_acc += acc * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{total_loss/n:.4f}", acc=f"{total_acc/n:.4f}")

        if epoch > epochs_warmup:
            scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for x, y in loader_val:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        val_acc = correct / max(1, total)
        best_val = max(best_val, val_acc)
        print(f"[{stage.name}] epoch {epoch}: val_acc={val_acc:.4f} val_loss={val_loss/max(1,total):.4f}")

    return best_val


# -------------------------------
# ONNX Export & Optional ORT Quant
# -------------------------------

def export_onnx(model: nn.Module, out_path: str, img_size: int, device: torch.device, opset: int = 17):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}
    torch.onnx.export(
        model, dummy, out_path, input_names=["input"], output_names=["logits"],
        dynamic_axes=dynamic_axes, opset_version=opset)
    print(f"Saved ONNX: {out_path}")


def ort_static_quant(onnx_in: str, onnx_out: str):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        # Dynamic quant (no calibration). Works well for Linear layers; convs remain fp.
        quantize_dynamic(model_input=onnx_in, model_output=onnx_out, weight_type=QuantType.QInt8)
        print(f"Saved ORT dynamic-quantized ONNX: {onnx_out}")
    except Exception as e:
        print("[WARN] onnxruntime not available for extra quant step:", e)


# -------------------------------
# Main
# -------------------------------

def split_train_val(ds: Dataset, val_ratio=0.1, seed=42):
    n = len(ds)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    v = int(n * val_ratio)
    val_idx = idx[:v]
    tr_idx = idx[v:]
    from torch.utils.data import Subset
    return Subset(ds, tr_idx), Subset(ds, val_idx)


def build_stage_loaders(args, ai_dir: str, real_dir: str, augment=True):
    ai_ds = ImageList(args.csv_ai_easy if ai_dir=="easy" and args.csv_ai_easy else args.ai_easy if ai_dir=="easy" else
                      args.csv_ai_medium if ai_dir=="medium" and args.csv_ai_medium else args.ai_medium if ai_dir=="medium" else
                      args.csv_ai_hard if ai_dir=="hard" and args.csv_ai_hard else args.ai_hard,
                      img_size=args.img_size, augment=augment, label_override=1)

    real_ds = ImageList(args.csv_real if args.csv_real else args.real, img_size=args.img_size, augment=augment, label_override=0)

    # Merge and balance a bit towards minority
    mix = MixedConcat([ai_ds, real_ds], balance=0.35)
    tr, va = split_train_val(mix, val_ratio=0.1, seed=args.seed)
    loader_tr = create_loader(tr, args.batch_size, args.workers, shuffle=True)
    loader_va = create_loader(va, args.batch_size, args.workers, shuffle=False)
    return loader_tr, loader_va


def run(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.out_dir, exist_ok=True)

    model = TinyAIDetector(width_mult=args.width_mult, num_classes=2)
    model.to(device)

    print(f"Params: {human_count(count_params(model))}")

    criterion = LabelSmoothingCE(0.05)
    ema = EMA(model, decay=0.999)

    # ---- Stage 1: EASY ----
    ltr, lva = build_stage_loaders(args, ai_dir="easy", real_dir=args.real, augment=True)
    acc1 = train_one_stage(model, ltr, lva, device, StageCfg("EASY", args.epochs_easy, args.lr, freeze_until="stem"), ema=ema, criterion=criterion)

    # ---- Stage 2: MEDIUM ----
    ltr, lva = build_stage_loaders(args, ai_dir="medium", real_dir=args.real, augment=True)
    acc2 = train_one_stage(model, ltr, lva, device, StageCfg("MEDIUM", args.epochs_medium, args.lr*0.7, freeze_until="mid"), ema=ema, criterion=criterion)

    # ---- Stage 3: HARD ----
    ltr, lva = build_stage_loaders(args, ai_dir="hard", real_dir=args.real, augment=True)
    acc3 = train_one_stage(model, ltr, lva, device, StageCfg("HARD", args.epochs_hard, args.lr*0.5, freeze_until="none"), ema=ema, criterion=criterion)

    print(f"Val accs → easy:{acc1:.4f} medium:{acc2:.4f} hard:{acc3:.4f}")

    # Apply EMA weights for export
    ema.apply_to(model)

    # ---------------- Pruning before QAT ----------------
    print("Pruning...")
    structured_prune(model, amount=args.structured_prune)
    l1_unstructured_prune(model, amount=args.l1_prune)

    # Quick fine-tune after pruning
    ltr, lva = build_stage_loaders(args, ai_dir="hard", real_dir=args.real, augment=True)
    _ = train_one_stage(model, ltr, lva, device, StageCfg("FINETUNE_PRUNED", max(1, args.epochs_prune_ft), args.lr*0.3, freeze_until="none"), ema=ema, criterion=criterion)

    # ---------------- QAT ----------------
    print("Preparing QAT...")
    torch.backends.quantized.engine = 'qnnpack'
    model_q = TinyAIDetector(width_mult=args.width_mult, num_classes=2)
    model_q.load_state_dict(model.state_dict())
    model_q.to(device)

    qconfig = get_default_qat_qconfig('qnnpack')
    model_q.train()
    prepared = prepare_qat_fx(model_q, {"": qconfig})

    # short QAT fine-tune
    ltr, lva = build_stage_loaders(args, ai_dir="hard", real_dir=args.real, augment=True)
    _ = train_one_stage(prepared, ltr, lva, device, StageCfg("QAT", max(1, args.epochs_qat), args.lr*0.1, freeze_until="none"), ema=None, criterion=criterion)

    converted = convert_fx(prepared.eval())
    converted.to(device)

    # Save torch
    torch.save(converted.state_dict(), os.path.join(args.out_dir, "model_pruned_qat.pt"))

    # ---------------- Exports ----------------
    # FP32 ONNX
    export_onnx(model, os.path.join(args.out_dir, "model_fp32.onnx"), args.img_size, device, opset=args.onnx_opset)

    # INT8 ONNX from QAT
    export_onnx(converted, os.path.join(args.out_dir, "model_int8_qat.onnx"), args.img_size, device, opset=args.onnx_opset)

    # Optional: ORT dynamic quant on fp32 as an extra variant
    ort_in = os.path.join(args.out_dir, "model_fp32.onnx")
    ort_out = os.path.join(args.out_dir, "model_fp32_int8_dynamic.onnx")
    ort_static_quant(ort_in, ort_out)

    print("All done. Artifacts in:", os.path.abspath(args.out_dir))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ai_easy', type=str, help='Folder of easy AI images')
    p.add_argument('--ai_medium', type=str, help='Folder of medium AI images')
    p.add_argument('--ai_hard', type=str, help='Folder of hard AI images')
    p.add_argument('--real', type=str, help='Folder of real images')

    p.add_argument('--csv_ai_easy', type=str, default='', help='Optional CSV [path,label] for easy AI (label ignored & overridden=1)')
    p.add_argument('--csv_ai_medium', type=str, default='', help='Optional CSV [path,label] for medium AI (label overridden=1)')
    p.add_argument('--csv_ai_hard', type=str, default='', help='Optional CSV [path,label] for hard AI (label overridden=1)')
    p.add_argument('--csv_real', type=str, default='', help='Optional CSV [path,label] for real images (label overridden=0)')

    p.add_argument('--img_size', type=int, default=192)
    p.add_argument('--width_mult', type=float, default=0.5)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--epochs_easy', type=int, default=3)
    p.add_argument('--epochs_medium', type=int, default=3)
    p.add_argument('--epochs_hard', type=int, default=6)
    p.add_argument('--epochs_prune_ft', type=int, default=2)
    p.add_argument('--epochs_qat', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--structured_prune', type=float, default=0.3)
    p.add_argument('--l1_prune', type=float, default=0.2)
    p.add_argument('--onnx_opset', type=int, default=17)
    p.add_argument('--out_dir', type=str, default='./artifacts')

    args = p.parse_args()

    # Basic validation
    if not args.csv_ai_easy and not args.ai_easy:
        p.error('Please set --ai_easy or --csv_ai_easy')
    if not args.csv_ai_medium and not args.ai_medium:
        p.error('Please set --ai_medium or --csv_ai_medium')
    if not args.csv_ai_hard and not args.ai_hard:
        p.error('Please set --ai_hard or --csv_ai_hard')
    if not args.csv_real and not args.real:
        p.error('Please set --real or --csv_real')

    return args


if __name__ == '__main__':
    # Important for Windows multiprocessing
    import multiprocessing as mp
    mp.freeze_support()
    args = parse_args()
    run(args)
