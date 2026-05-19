"""
Training script for RDD-COD
Triple-stream architecture: RGB-CNN (Res2Net-50) + RGB-Transformer (PVT-v2-b2) + Depth (PVT-v2-b1)
"""

import os
import csv
import torch
import numpy as np
from torch.amp import autocast
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime

import py_sod_metrics

#from datasets.camo_cod10k_dataset import DatasetWithDepth
from datasets.mhcd_dataset import DatasetWithDepth
from models.rdd_cod import RDD_COD
from loss.cod_loss import CODLoss

from utils.logger import setup_logger
from utils.plot import plot_training_curves


# ===================== Configuration =====================

class Config:
    def __init__(self):
        # ── Model ──────────────────────────────────────────────────────────
        self.n_classes  = 1
        self.pretrained = True

        # ── Dataset ────────────────────────────────────────────────────────
        #self.root      = "../Datasets/camo_cod10k"
        #self.dataset   = "CAMO_COD10K"
        self.root = "../Datasets/CUINet"
        self.dataset = "CAMO_COD10K_CUINet"
        self.img_size  = 352
        self.use_depth = True

        # ── Training ───────────────────────────────────────────────────────
        self.epochs      = 120
        self.batch_size  = 8
        self.num_workers = 4

        # ── Learning rates ─────────────────────────────────────────────────
        self.lr_cnn     = 5e-5
        self.lr_trans   = 5e-5
        self.lr_depth   = 1e-4
        self.lr_fusion  = 2e-4
        self.lr_decoder = 2e-4
        self.weight_decay = 1e-4

        # ── Loss weights ───────────────────────────────────────────────────
        self.seg_weights       = (1.0, 0.8, 0.6, 0.4)  # d1..d4
        self.lambda_edge       = 0.5    # boundary supervision
        self.lambda_depth_edge = 0.3    # depth Sobel edge supervision
        self.lambda_entropy    = 0.01   # CDFM fusion weight entropy regularization

        # ── Warmup schedule ────────────────────────────────────────────────
        self.warmup_epochs          = 5   # Phase 1: tất cả encoder bị freeze
        self.warmup_rgb_epochs      = 15  # Phase 2: depth encoder bị freeze
        self.warmup_boundary_epochs = 10  # boundary + depth-edge loss bắt đầu sau epoch này

        # ── Scheduler ──────────────────────────────────────────────────────
        self.use_cosine_schedule = True
        self.min_lr = 1e-6

        # ── Device ─────────────────────────────────────────────────────────
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Resume ─────────────────────────────────────────────────────────
        self.resume     = False
        self.resume_dir = ""

        # ── Logging ────────────────────────────────────────────────────────
        if self.resume and self.resume_dir:
            self.log_dir = self.resume_dir
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_dir = f"logs/RDD_COD_{self.dataset}_{ts}"
        os.makedirs(self.log_dir, exist_ok=True)


# ===================== pysodmetrics helpers =====================

def tensor_to_numpy_uint8(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(1)
    elif tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().numpy()
    return (arr * 255).clip(0, 255).astype(np.uint8)


def compute_metrics(pred_list_np, gt_list_np):
    SM    = py_sod_metrics.Smeasure()
    EM    = py_sod_metrics.Emeasure()
    WFM   = py_sod_metrics.WeightedFmeasure()
    FM    = py_sod_metrics.Fmeasure()
    MAE_m = py_sod_metrics.MAE()

    for pred, gt in zip(pred_list_np, gt_list_np):
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        FM.step(pred=pred, gt=gt)
        MAE_m.step(pred=pred, gt=gt)

    sm  = SM.get_results()["sm"]
    em  = EM.get_results()["em"]
    wfm = WFM.get_results()["wfm"]
    fm  = FM.get_results()["fm"]
    mae = MAE_m.get_results()["mae"]

    return {
        "Sm":     sm,
        "meanEm": em["curve"].mean(),
        "maxEm":  em["curve"].max(),
        "adpEm":  em["adp"],
        "wFm":    wfm,
        "meanFm": fm["curve"].mean(),
        "maxFm":  fm["curve"].max(),
        "adpFm":  fm["adp"],
        "MAE":    mae,
    }


# ===================== Training =====================

def train_epoch(model, loader, optimizer, criterion, scaler, device, config, epoch):
    model.train()

    # ── Warmup freeze strategy ────────────────────────────────────────────
    # Phase 1: chỉ train decoder, toàn bộ encoder bị freeze
    # Phase 2: RGB encoders được unfreeze, depth encoder vẫn frozen
    # Phase 3: toàn bộ model được unfreeze
    if epoch <= config.warmup_epochs:
        for name, param in model.named_parameters():
            param.requires_grad = ('encoder' not in name)
    elif epoch <= config.warmup_rgb_epochs:
        for name, param in model.named_parameters():
            if 'depth_backbone' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    # Boundary + depth-edge loss chỉ kích hoạt sau warmup_boundary_epochs
    use_boundary = (epoch > config.warmup_boundary_epochs)

    total_loss = 0.0
    loss_accum = {'seg_loss': 0.0, 'edge_loss': 0.0,
                  'depth_edge_loss': 0.0, 'entropy_loss': 0.0}
    num_batches = 0

    for rgb, depth, masks in loader:
        rgb   = rgb.to(device)
        depth = depth.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16):
            predictions, edge_out, depth_edge_map, fusion_weights = model(rgb, depth)

            if use_boundary:
                loss, loss_dict = criterion(
                    predictions, edge_out, depth_edge_map,
                    fusion_weights, masks, gt_edge=None
                )
            else:
                # Tạm tắt boundary-related losses trong giai đoạn warmup
                loss, loss_dict = criterion(
                    predictions, edge_out, depth_edge_map,
                    fusion_weights, masks, gt_edge=None,
                    override_lambdas={'lambda_edge': 0.0, 'lambda_depth_edge': 0.0}
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k in loss_accum:
            loss_accum[k] += loss_dict.get(k, 0.0)
        num_batches += 1

    result = {'total': total_loss / num_batches}
    result.update({k: v / num_batches for k, v in loss_accum.items()})
    return result


@torch.no_grad()
def validate(model, loader, criterion, device, config):
    model.eval()

    total_loss  = 0.0
    num_samples = 0
    all_preds, all_gts = [], []

    for rgb, depth, masks in loader:
        rgb   = rgb.to(device)
        depth = depth.to(device)
        masks = masks.to(device)

        predictions, edge_out, depth_edge_map, fusion_weights = model(rgb, depth)

        loss, _ = criterion(
            predictions, edge_out, depth_edge_map,
            fusion_weights, masks, gt_edge=None
        )

        pred_probs = torch.sigmoid(predictions[0])   # d1 = finest scale

        batch_size  = rgb.shape[0]
        total_loss  += loss.item() * batch_size
        num_samples += batch_size

        pred_np = tensor_to_numpy_uint8(pred_probs)
        gt_np   = tensor_to_numpy_uint8(masks)
        for i in range(batch_size):
            all_preds.append(pred_np[i])
            all_gts.append(gt_np[i])

    metrics = compute_metrics(all_preds, all_gts)
    metrics["loss"] = total_loss / num_samples
    return metrics


# ===================== Optimizer / Scheduler =====================

def create_optimizer(model, config):
    """
    Tạo optimizer với learning rate riêng biệt cho từng nhóm tham số:
      - cnn_backbone:   pretrained Res2Net-50, lr thấp
      - trans_backbone: pretrained PVT-v2-b2,  lr thấp
      - depth_backbone: pretrained PVT-v2-b1,  lr trung bình
      - fusion:         TripleCDFM + BAM,       lr cao
      - decoder:        FEM + output heads,     lr cao
    """
    cnn_p, trans_p, depth_p, fusion_p, decoder_p = [], [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'cnn_backbone' in name:
            cnn_p.append(param)
        elif 'trans_backbone' in name:
            trans_p.append(param)
        elif 'depth_backbone' in name:
            depth_p.append(param)
        elif any(k in name for k in ('cdfm', 'bam', 'depth_gate')):
            fusion_p.append(param)
        else:
            decoder_p.append(param)

    groups = []
    for params, lr, name in [
        (cnn_p,     config.lr_cnn,     'cnn_backbone'),
        (trans_p,   config.lr_trans,   'trans_backbone'),
        (depth_p,   config.lr_depth,   'depth_backbone'),
        (fusion_p,  config.lr_fusion,  'fusion'),
        (decoder_p, config.lr_decoder, 'decoder'),
    ]:
        if params:
            groups.append({'params': params, 'lr': lr, 'name': name})
            print(f"  {name}: {len(params)} params, lr={lr:.2e}")

    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def create_scheduler(optimizer, config, steps_per_epoch):
    if config.use_cosine_schedule:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs * steps_per_epoch, eta_min=config.min_lr)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)


# ===================== Main =====================

def main():
    config = Config()
    logger = setup_logger(config.log_dir, "train.log")

    logger.info("=" * 80)
    logger.info("RDD-COD TRAINING")
    logger.info("=" * 80)
    logger.info("Loss configuration:")
    logger.info(f"  lambda_edge:        {config.lambda_edge}")
    logger.info(f"  lambda_depth_edge:  {config.lambda_depth_edge}")
    logger.info(f"  lambda_entropy:     {config.lambda_entropy}")
    logger.info("=" * 80)

    # Datasets
    logger.info("Loading datasets...")
    train_dataset = DatasetWithDepth(root=config.root, split="train",
                                     img_size=config.img_size, augment=True,
                                     use_depth=config.use_depth, logger=logger)
    val_dataset   = DatasetWithDepth(root=config.root, split="val",
                                     img_size=config.img_size, augment=False,
                                     use_depth=config.use_depth, logger=logger)
    logger.info(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True,  num_workers=config.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers, pin_memory=True)

    # Model
    logger.info("Creating model...")
    model = RDD_COD(
        n_classes=config.n_classes, pretrained=config.pretrained
    ).to(config.device)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_p:,}  Trainable: {trainable_p:,}")

    # Loss
    criterion = CODLoss(
        seg_weights       = config.seg_weights,
        lambda_edge       = config.lambda_edge,
        lambda_depth_edge = config.lambda_depth_edge,
        lambda_entropy    = config.lambda_entropy,
    )

    # Optimizer / scheduler
    logger.info("Creating optimizer...")
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    scaler    = torch.amp.GradScaler('cuda')

    # Training state
    start_epoch         = 1
    best_s_measure      = 0.0
    train_losses        = []
    val_losses          = []
    val_metrics_history = []

    # Resume từ checkpoint
    if config.resume and config.resume_dir:
        ckpt_path = os.path.join(config.resume_dir, "last.pth")
        if os.path.exists(ckpt_path):
            logger.info(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=config.device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch         = ckpt["epoch"] + 1
            best_s_measure      = ckpt.get("best_s_measure", 0.0)
            train_losses        = ckpt.get("train_losses", [])
            val_losses          = ckpt.get("val_losses", [])
            val_metrics_history = ckpt.get("val_metrics", [])
            logger.info(f"  Resumed from epoch {start_epoch}, best Sm: {best_s_measure:.4f}")

    # CSV header
    csv_path = os.path.join(config.log_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "seg_loss", "edge_loss",
                "depth_edge_loss", "entropy_loss",
                "val_loss", "Sm", "meanEm", "maxEm", "wFm", "MAE"
            ])

    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    for epoch in range(start_epoch, config.epochs + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"EPOCH {epoch}/{config.epochs}")
        logger.info(f"{'=' * 60}")

        if epoch <= config.warmup_epochs:
            logger.info("[PHASE 1] All encoders frozen")
        elif epoch <= config.warmup_rgb_epochs:
            logger.info("[PHASE 2] RGB encoders unfrozen, depth frozen")
        elif epoch == config.warmup_rgb_epochs + 1:
            logger.info("[PHASE 3] All encoders unfrozen")

        if epoch <= config.warmup_boundary_epochs:
            logger.info("[WARMUP] Boundary + depth-edge loss disabled")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 scaler, config.device, config, epoch)
        train_losses.append(train_loss['total'])

        # Validate
        val_metrics = validate(model, val_loader, criterion, config.device, config)
        val_losses.append(val_metrics["loss"])
        val_metrics_history.append(val_metrics)

        # Log
        logger.info(
            f"[TRAIN] total={train_loss['total']:.4f} | "
            f"seg={train_loss['seg_loss']:.4f} | "
            f"edge={train_loss['edge_loss']:.4f} | "
            f"depth_edge={train_loss['depth_edge_loss']:.4f} | "
            f"entropy={train_loss['entropy_loss']:.4f}"
        )
        logger.info(
            f"[VAL]   loss={val_metrics['loss']:.4f} | "
            f"Sm={val_metrics['Sm']:.4f} | "
            f"maxEm={val_metrics['maxEm']:.4f} | "
            f"wFm={val_metrics['wFm']:.4f} | "
            f"MAE={val_metrics['MAE']:.4f}"
        )

        # CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                train_loss['total'], train_loss['seg_loss'], train_loss['edge_loss'],
                train_loss['depth_edge_loss'], train_loss['entropy_loss'],
                val_metrics["loss"], val_metrics["Sm"],
                val_metrics["meanEm"], val_metrics["maxEm"],
                val_metrics["wFm"], val_metrics["MAE"]
            ])

        # Checkpoint
        ckpt = {
            "epoch":          epoch,
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "scaler":         scaler.state_dict(),
            "best_s_measure": best_s_measure,
            "train_losses":   train_losses,
            "val_losses":     val_losses,
            "val_metrics":    val_metrics_history,
            "config":         vars(config),
        }
        torch.save(ckpt, os.path.join(config.log_dir, "last.pth"))

        if val_metrics["Sm"] > best_s_measure:
            best_s_measure = val_metrics["Sm"]
            torch.save(model.state_dict(), os.path.join(config.log_dir, "best.pth"))
            logger.info(f"NEW BEST Sm: {best_s_measure:.4f}")

        # Scheduler step
        if config.use_cosine_schedule:
            for _ in range(len(train_loader)):
                scheduler.step()
        else:
            scheduler.step()

        if epoch % 5 == 0 or epoch == config.epochs:
            plot_training_curves(train_losses, val_losses, val_metrics_history, config.log_dir)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best Sm: {best_s_measure:.4f}")
    logger.info(f"Model saved to: {config.log_dir}")


if __name__ == "__main__":
    main()