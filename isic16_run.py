#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISIC18 Train & Predict (single-file, resumable & split-persistent)

Features
- Single-run training (no k-fold)
- Stable Train/Val/Test split with on-disk persistence (splits.json)
- Early stopping with state resume (best & counter)
- Robust checkpointing: best.pth / last.pth (+ optimizer, early state)
- History persistence & append (history.json) + continuous plots
- AMP mixed precision, ReduceLROnPlateau, grad clipping
- Predict supports --split train|val|test|auto, reusing saved splits
"""

import os
import json
import math
import time
import random
import argparse
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from tqdm.auto import tqdm, trange

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- your project imports (保持你当前工程的相对路径) ----
from Dataloader_ISIC16 import PairedData     # 需确保：统一 resize；image->[C,H,W] float；mask->[H,W] long
from loss import dice_loss                   # 返回 (loss=1-avg_dice, per_class_dice)

from RS3Mamba.UNetFormer import UNetFormer
from Network.SwinUNet import SwinTransformerSys
from Network.UNet import UNet
from Network.transunet import TransUNet
from Network.UNetPP import NestedUNet
from ResNet.ResNet import resnet18
from UKAN.archs import UKAN
from Network.attenUnet import AttentionUNet
from UTNet.utnet import UTNet
from Network.malunet import MALUNet
from pytorch_dcsaunet.DCSAU_Net import dcsaunet
from Network.egeunet import EGEUNet


# ------------------------------
# Utils
# ------------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_once(msg: str, f=None):
    print(msg)
    if f is not None:
        f.write(msg + "\n")
        f.flush()


def save_plot(x, y1, y2, out_dir, metric_name="Loss"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(x, y1, marker='o', label=f"Train {metric_name}")
    plt.plot(x, y2, marker='o', label=f"Val {metric_name}")
    plt.title(f"{metric_name} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{metric_name.lower().replace(' ', '_')}_curve.png"))
    plt.close()


def safe_collate(batch):
    # 规避 torchvision.datapoints 或不可 resize storage 的坑
    imgs, masks = zip(*batch)
    imgs  = [torch.as_tensor(i).contiguous().clone() for i in imgs]
    masks = [torch.as_tensor(m).contiguous().clone() for m in masks]
    return torch.stack(imgs, dim=0), torch.stack(masks, dim=0)


# ------------------------------
# Early Stopping (可恢复)
# ------------------------------
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0, best: Optional[float]=None, counter: int=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = best
        self.counter = counter
        self.should_stop = False

    def step(self, current):
        if self.best is None or (self.best - current) > self.min_delta:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ------------------------------
# Checkpoint & History & Splits
# ------------------------------
def save_checkpoint(state: Dict[str, Any], out_dir: str, is_best: bool):
    os.makedirs(out_dir, exist_ok=True)
    last_path = os.path.join(out_dir, "last.pth")
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(out_dir, "best.pth")
        torch.save(state, best_path)


def load_checkpoint(model, optimizer, ckpt_path: str):
    if not (ckpt_path and os.path.exists(ckpt_path)):
        return 0, math.inf, None, 0
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val = ckpt.get("best_val", math.inf)
    early_best = ckpt.get("early_best", best_val)
    early_counter = ckpt.get("early_counter", 0)
    print(f"[Resume] Loaded {ckpt_path} at epoch={start_epoch}, best_val={best_val:.6f} (early best={early_best:.6f}, counter={early_counter})")
    return start_epoch, best_val, early_best, early_counter


def load_history(path: str) -> Dict[str, List[float]]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"epochs": [], "train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}


def append_history(path: str, epoch: int, train_loss: float, val_loss: float):
    hist = load_history(path)
    hist["epochs"].append(int(epoch))
    hist["train_loss"].append(float(train_loss))
    hist["val_loss"].append(float(val_loss))
    hist["train_dice"].append(float(1.0 - train_loss))
    hist["val_dice"].append(float(1.0 - val_loss))
    with open(path, "w") as f:
        json.dump(hist, f, indent=2)
    return hist


def exists_split(root: str, split: str) -> bool:
    img_dir = os.path.join(root, split, "image")
    return os.path.isdir(img_dir) and len(os.listdir(img_dir)) > 0


def _subset_indices(ds):
    return ds.indices if isinstance(ds, Subset) else None


def _base_dataset(ds):
    return ds.dataset if isinstance(ds, Subset) else ds


def save_splits(out_dir: str, train_ds, val_ds, test_ds, dataset_root: str, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    base = _base_dataset(train_ds)
    img_list = getattr(base, "image_path", None)
    payload = {
        "train_indices": _subset_indices(train_ds),
        "val_indices": _subset_indices(val_ds),
        "test_indices": _subset_indices(test_ds),
        "test_from": "dir" if isinstance(test_ds, PairedData) and getattr(test_ds, "target", "") == "test" else ("split" if isinstance(test_ds, Subset) else "unknown"),
        "seed": seed,
    }
    if img_list:
        # 存储基数据集的文件基名，便于将来一致性检查
        payload["base_filenames"] = [os.path.basename(p) for p in img_list]
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Split] saved to {os.path.join(out_dir, 'splits.json')}")


def try_load_splits(path: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# ------------------------------
# Model init (merged)
# ------------------------------
def initialize_model(model_name: str, device, num_classes=2, img_size=256, in_chans=3, pretrained=False):
    name = model_name.strip()
    if name == "UNet":
        model = UNet(in_channel=in_chans, out_channel=num_classes)
    elif name == "TransUNet":
        model = TransUNet(img_dim=img_size, in_channels=in_chans, out_channels=128,
                          head_num=4, mlp_dim=512, block_num=8, patch_dim=16, class_num=num_classes)
    elif name == "SwinUNet":
        model = SwinTransformerSys(img_size=img_size, patch_size=8, in_chans=in_chans,
                                   num_classes=num_classes, embed_dim=96,
                                   depths=[2,2,2,2], num_heads=[3,6,12,24], window_size=4, mlp_ratio=4.0)
    elif name == "UNetPP":
        model = NestedUNet(num_classes=num_classes, input_channels=in_chans, deep_supervision=False)
    elif name == "ResNet18":
        model = resnet18(pretrained=pretrained, num_classes=num_classes)
    elif name == "UKAN":
        model = UKAN(num_classes=num_classes, input_channels=in_chans, img_size=img_size, patch_size=1)
    elif name == "UNetFormer":
        model = UNetFormer(num_classes=num_classes, pretrained=False)
    elif name == "AttentionUNet":
        model = AttentionUNet(output_ch=num_classes)
    elif name == "UTNet":
        model = UTNet(in_chan=in_chans,num_classes=num_classes,base_chan=32)
    elif name == "MALUNet":
        model = MALUNet(num_classes=num_classes,input_channels=in_chans)
    elif name == "DCSAUNet":
        model = dcsaunet(img_channels=in_chans,n_classes=num_classes)
    elif name == "EGEUNet":
        model = EGEUNet(num_classes=num_classes, input_channels=in_chans, gt_ds=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)


# ------------------------------
# Data split helpers (支持从 splits.json 恢复)
# ------------------------------
def build_dataloaders(
    dataset_root: str,
    batch_size: int,
    img_size: int,
    in_chans: int,
    num_workers: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    saved_splits: Optional[Dict[str, Any]] = None
):
    g = torch.Generator(); g.manual_seed(seed)

    # 1) 如有 splits.json，优先使用
    if saved_splits is not None and saved_splits.get("train_indices") is not None and saved_splits.get("val_indices") is not None:
        full_train = PairedData(root=dataset_root, target="train", img_size=(img_size, img_size), rgb=(in_chans==3))
        train_ds = Subset(full_train, saved_splits["train_indices"])
        val_ds   = Subset(full_train, saved_splits["val_indices"])

        if saved_splits.get("test_from") == "split" and saved_splits.get("test_indices") is not None:
            test_ds = Subset(full_train, saved_splits["test_indices"])
        elif exists_split(dataset_root, "test"):
            test_ds = PairedData(root=dataset_root, target="test", img_size=(img_size, img_size), rgb=(in_chans==3))
        else:
            # 若没保存 test_indices 且也没有 test 目录，则用补集
            all_indices = set(range(len(full_train)))
            used = set(saved_splits["train_indices"]) | set(saved_splits["val_indices"])
            test_idx = sorted(list(all_indices - used))
            test_ds = Subset(full_train, test_idx)

    else:
        # 2) 正常流程：若存在 test 目录，则 train 里再切分 val；否则在 train 里按比例切 train/val/test
        if exists_split(dataset_root, "test"):
            full_train = PairedData(root=dataset_root, target="train", img_size=(img_size, img_size), rgb=(in_chans==3))
            train_len = len(full_train)
            val_len = max(1, int(train_len * val_ratio))
            train_len2 = max(1, train_len - val_len)
            train_ds, val_ds = torch.utils.data.random_split(full_train, [train_len2, val_len], generator=g)
            test_ds = PairedData(root=dataset_root, target="test", img_size=(img_size, img_size), rgb=(in_chans==3))
        else:
            full = PairedData(root=dataset_root, target="train", img_size=(img_size, img_size), rgb=(in_chans==3))
            n = len(full)
            t = max(1, int(n * train_ratio))
            v = max(1, int(n * val_ratio))
            t = min(t, n-2)  # 至少给 val/test 留空间
            v = min(v, n-t-1)
            te = n - t - v
            train_ds, val_ds, test_ds = torch.utils.data.random_split(full, [t, v, te], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers>0), collate_fn=safe_collate)

    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers>0), collate_fn=safe_collate)

    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers>0), collate_fn=safe_collate)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


# ------------------------------
# Train / Validate
# ------------------------------
def train_one_epoch(model, loader, optimizer, device, scaler, clip_grad=None):
    model.train()
    run_loss = 0.0; n_items = 0
    pbar = tqdm(loader, desc="Train", leave=False, dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(images)
            loss, _ = dice_loss(outputs, labels)  # loss = 1 - dice
        scaler.scale(loss).backward()
        if clip_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer); scaler.update()

        bs = images.size(0)
        run_loss += loss.item() * bs; n_items += bs
        cur_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{run_loss/max(1,n_items):.4f}", lr=f"{cur_lr:.2e}")
    return run_loss / max(1, n_items)


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    run_loss = 0.0; n_items = 0
    pbar = tqdm(loader, desc="Val", leave=False, dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss, _ = dice_loss(outputs, labels)
        bs = images.size(0)
        run_loss += loss.item() * bs; n_items += bs
        pbar.set_postfix(val_loss=f"{run_loss/max(1,n_items):.4f}")
    return run_loss / max(1, n_items)


# ------------------------------
# Predict / Evaluate
# ------------------------------
@torch.no_grad()
def predict_and_evaluate(model, loader, device, out_dir, num_classes=2, include_background=False):
    import numpy as np
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    total_dice = np.zeros(num_classes, dtype=np.float64)
    total_iou  = np.zeros(num_classes, dtype=np.float64)
    total_pre  = np.zeros(num_classes, dtype=np.float64)
    total_rec  = np.zeros(num_classes, dtype=np.float64)
    counts     = np.zeros(num_classes, dtype=np.int64)

    idx_global = 0
    pbar = tqdm(loader, desc="Predict/Test", dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device); labels = labels.to(device)
        logits = model(images)
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)

        B = images.size(0)
        for i in range(B):
            pred = preds[i].cpu().numpy().astype(np.int32)
            lab  = labels[i].cpu().numpy().astype(np.int32)

            Image.fromarray(pred.astype(np.uint8), mode='L').save(
                os.path.join(out_dir, f"pred_{idx_global+i:06d}.png")
            )

            for c in range(num_classes):
                if not include_background and c == 0:
                    continue
                pred_c = (pred == c); lab_c = (lab == c)
                inter = np.logical_and(pred_c, lab_c).sum()
                union = np.logical_or(pred_c, lab_c).sum()
                p_sum = pred_c.sum(); l_sum = lab_c.sum()

                if union == 0 and l_sum == 0 and p_sum == 0:
                    continue

                dice = (2*inter) / (p_sum + l_sum + 1e-6)
                iou  = inter / (union + 1e-6)
                prec = inter / (p_sum + 1e-6)
                rec  = inter / (l_sum + 1e-6)

                total_dice[c] += dice; total_iou[c] += iou
                total_pre[c]  += prec; total_rec[c] += rec
                counts[c]     += 1

        idx_global += B

    avg_dice = np.where(counts>0, total_dice/np.maximum(1,counts), 0)
    avg_iou  = np.where(counts>0, total_iou /np.maximum(1,counts), 0)
    avg_pre  = np.where(counts>0, total_pre /np.maximum(1,counts), 0)
    avg_rec  = np.where(counts>0, total_rec /np.maximum(1,counts), 0)

    c_range = range(num_classes) if include_background else range(1, num_classes)
    overall = {
        "dice": float(np.mean([avg_dice[c] for c in c_range]) if len(list(c_range))>0 else 0.0),
        "iou":  float(np.mean([avg_iou[c]  for c in c_range]) if len(list(c_range))>0 else 0.0),
        "precision": float(np.mean([avg_pre[c] for c in c_range]) if len(list(c_range))>0 else 0.0),
        "recall":    float(np.mean([avg_rec[c] for c in c_range]) if len(list(c_range))>0 else 0.0),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "per_class": {str(c): {
                "dice": float(avg_dice[c]), "iou": float(avg_iou[c]),
                "precision": float(avg_pre[c]), "recall": float(avg_rec[c]),
                "count": int(counts[c]),
            } for c in range(num_classes)},
            "overall": overall,
            "include_background": include_background
        }, f, indent=2)

    return avg_dice, avg_iou, overall


# ------------------------------
# Main Train loop (支持 resume 的 splits/history/early)
# ------------------------------
def run_train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, "train.log"), "a")

    # ——尝试从 resume 目录读取 splits.json——
    saved_splits = None
    if args.resume:
        resume_dir = os.path.dirname(args.resume)
        saved_splits = try_load_splits(os.path.join(resume_dir, "splits.json"))

    # dataloaders（若有保存的划分则复用）
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        in_chans=args.in_chans,
        num_workers=args.workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        saved_splits=saved_splits
    )

    # model/opt/sched
    model = initialize_model(args.model, device, num_classes=args.num_classes,
                             img_size=args.img_size, in_chans=args.in_chans,
                             pretrained=args.pretrained)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience, min_lr=args.min_lr, verbose=True
    )
    scaler = GradScaler(enabled=args.amp)

    # ——恢复训练状态（epoch、best、early）——
    start_epoch, best_val, early_best, early_counter = load_checkpoint(model, optimizer, args.resume) if args.resume else (0, math.inf, None, 0)
    early = EarlyStopping(patience=args.patience, min_delta=args.min_delta, best=early_best, counter=early_counter)

    # ——载入或初始化 history（保证曲线延续）——
    hist_path = os.path.join(out_dir, "history.json")
    history = load_history(hist_path)
    if len(history["epochs"]) > 0:
        print_once(f"[History] Loaded {len(history['epochs'])} entries from history.json", log_f)

    if not args.resume:
        print_once(f"[Start] model={args.model}, img_size={args.img_size}, in_chans={args.in_chans}, "
                   f"num_classes={args.num_classes}, seed={args.seed}", log_f)

    # ——若无 splits.json（新训练或其它目录），保存当前划分——
    if saved_splits is None:
        save_splits(out_dir, train_ds, val_ds, test_ds, args.dataset_root, args.seed)

    # ——训练循环（从 start_epoch 接续）——
    for epoch in trange(start_epoch, args.epochs, desc="Epoch", dynamic_ncols=True):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, clip_grad=args.clip_grad)
        val_loss   = validate_one_epoch(model, val_loader, device)

        # ——历史记录：追加并重画（不会丢失之前的 1..start_epoch-1）——
        history = append_history(hist_path, epoch+1, train_loss, val_loss)
        save_plot(history["epochs"], history["train_loss"], history["val_loss"], out_dir, metric_name="Loss")
        save_plot(history["epochs"], history["train_dice"], history["val_dice"], out_dir, metric_name="Dice Coefficient")

        # ——调度与保存——
        scheduler.step(val_loss)
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        save_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val": best_val,
            "val_loss": val_loss,
            # 早停状态持久化
            "early_best": early.best if early.best is not None else best_val,
            "early_counter": early.counter
        }, out_dir, is_best=is_best)

        print_once(f"[Epoch {epoch+1:03d}/{args.epochs}] "
                   f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                   f"val_dice={1.0-val_loss:.4f} time={(time.time()-t0):.1f}s", log_f)

        # ——早停判定（延续历史）——
        if early.step(val_loss):
            print_once(f"[EarlyStop] patience={args.patience}, best_val={best_val:.6f} "
                       f"(early_best={early.best:.6f}, counter={early.counter})", log_f)
            break

    log_f.close()

    # ——使用 best.pth 评估 test——
    best_ckpt = os.path.join(out_dir, "best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location="cpu")["model_state_dict"])
    test_out_dir = os.path.join(out_dir, "test_pred")
    _, _, overall = predict_and_evaluate(
        model, test_loader, device, test_out_dir,
        num_classes=args.num_classes, include_background=args.include_background
    )
    print(f"[Test] overall dice={overall['dice']:.4f} iou={overall['iou']:.4f} -> saved to {test_out_dir}")


def run_predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ——优先用模型目录的 splits.json——
    model_dir = os.path.dirname(args.checkpoint)
    saved_splits = try_load_splits(os.path.join(model_dir, "splits.json"))

    # 选择 split
    split = args.split
    if split == "auto":
        if saved_splits is not None:
            # 若训练时有 test_from=dir，就用 test；否则用 split 的 test_indices；再不然回到 train
            if saved_splits.get("test_from") == "dir" and exists_split(args.dataset_root, "test"):
                split = "test"
            elif saved_splits.get("test_indices") is not None:
                split = "test"  # 来自划分
            else:
                split = "train"
        else:
            split = "test" if exists_split(args.dataset_root, "test") else "train"

    # 构建 dataset
    if split == "test" and exists_split(args.dataset_root, "test"):
        base_ds = PairedData(args.dataset_root, target="test",
                             img_size=(args.img_size, args.img_size), rgb=(args.in_chans==3))
        test_ds = base_ds
        print("[Predict] Using physical test/ directory.")
    else:
        base_ds = PairedData(args.dataset_root, target="train",
                             img_size=(args.img_size, args.img_size), rgb=(args.in_chans==3))
        if saved_splits is not None:
            if split == "train" and saved_splits.get("train_indices") is not None:
                test_ds = Subset(base_ds, saved_splits["train_indices"])
                print(f"[Predict] Using saved train split: {len(saved_splits['train_indices'])} samples.")
            elif split == "val" and saved_splits.get("val_indices") is not None:
                test_ds = Subset(base_ds, saved_splits["val_indices"])
                print(f"[Predict] Using saved val split: {len(saved_splits['val_indices'])} samples.")
            elif split == "test" and saved_splits.get("test_indices") is not None:
                test_ds = Subset(base_ds, saved_splits["test_indices"])
                print(f"[Predict] Using saved test split: {len(saved_splits['test_indices'])} samples.")
            else:
                test_ds = base_ds
                print(f"[Predict] Saved splits not found for '{split}', falling back to full '{base_ds.target}'.")
        else:
            test_ds = base_ds
            print(f"[Predict] No saved splits; using full '{base_ds.target}' directory.")

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True,
                             persistent_workers=(args.workers>0), collate_fn=safe_collate)

    # 模型与权重
    model = initialize_model(args.model, device, num_classes=args.num_classes,
                             img_size=args.img_size, in_chans=args.in_chans)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    out_dir = args.out_dir if args.out_dir else os.path.join(model_dir, f"pred_{split}")
    _, _, overall = predict_and_evaluate(
        model, test_loader, device, out_dir,
        num_classes=args.num_classes, include_background=args.include_background
    )
    print(f"[Predict] [{split}] overall dice={overall['dice']:.4f} iou={overall['iou']:.4f} -> saved to {out_dir}")


# ------------------------------
# CLI
# ------------------------------
def build_argparser():
    p = argparse.ArgumentParser("ISIC18 Train & Predict (single-file, resumable)")
    sub = p.add_subparsers(dest="mode", required=True)

    # train
    pt = sub.add_parser("train")
    pt.add_argument("--dataset-root", type=str, default="/home/jgzn/PycharmProjects/RZ/danzi/seg/datasets/ISIC16")
    pt.add_argument("--model", type=str, default="UNet")
    pt.add_argument("--num-classes", type=int, default=2)
    pt.add_argument("--img-size", type=int, default=256)
    pt.add_argument("--in-chans", type=int, default=3, help="3 for RGB, 1 for grayscale")
    pt.add_argument("--batch-size", type=int, default=16)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--epochs", type=int, default=100)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--weight-decay", type=float, default=1e-4)
    pt.add_argument("--min-lr", type=float, default=1e-6)
    pt.add_argument("--lr-patience", type=int, default=6)
    pt.add_argument("--clip-grad", type=float, default=1.0)
    pt.add_argument("--amp", action="store_true", default=True)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--out-dir", type=str, default="./results/ISIC16")
    pt.add_argument("--resume", type=str, default="", help="path to best/last.pth to resume")
    pt.add_argument("--pretrained", action="store_true", default=False)

    # split ratios (仅当不存在物理 test 目录且未提供 splits.json 时生效)
    pt.add_argument("--train-ratio", type=float, default=0.8)
    pt.add_argument("--val-ratio", type=float, default=0.1)
    pt.add_argument("--test-ratio", type=float, default=0.1)

    # early stopping
    pt.add_argument("--patience", type=int, default=10)
    pt.add_argument("--min-delta", type=float, default=0.0)

    # metrics
    pt.add_argument("--include-background", action="store_true", default=False)

    # predict
    pp = sub.add_parser("predict")
    pp.add_argument("--dataset-root", type=str, required=True)
    pp.add_argument("--model", type=str, default="UNet")
    pp.add_argument("--checkpoint", type=str, required=True)
    pp.add_argument("--num-classes", type=int, default=2)
    pp.add_argument("--img-size", type=int, default=256)
    pp.add_argument("--in-chans", type=int, default=3)
    pp.add_argument("--batch-size", type=int, default=1)
    pp.add_argument("--workers", type=int, default=2)
    pp.add_argument("--include-background", action="store_true", default=False)
    pp.add_argument("--out-dir", type=str, default="")
    pp.add_argument("--split", type=str, default="auto",
                    choices=["auto","train","val","test"],
                    help="Which split to predict on. 'auto' prefers saved splits; else test/ or train/")

    return p


def main():
    args = build_argparser().parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "predict":
        run_predict(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
