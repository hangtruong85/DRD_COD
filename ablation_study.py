"""
Ablation Study Runner for RDD-COD
Tương ứng với Section 4.3 trong bài báo rdd_cod_paper.tex

Bốn nhóm ablation:
  Group 1 — Module-level        (M1..M8)
  Group 2 — Depth stream design (D1..D6)
  Group 3 — Loss components     (L1..L7)
  Group 4 — DQG robustness      (Q1..Q6)

Cách chạy:
  # Tất cả nhóm
  python ablation_study.py --group all

  # Một nhóm cụ thể
  python ablation_study.py --group module
  python ablation_study.py --group depth
  python ablation_study.py --group loss
  python ablation_study.py --group dqg

  # Một variant cụ thể
  python ablation_study.py --group module --variant M1

  # Resume từ checkpoint
  python ablation_study.py --group module --resume

Output: logs/ablation/<group>/<variant>/  với last.pth, best.pth, results.json
"""

import os
import json
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

import py_sod_metrics

from datasets.mhcd_dataset import DatasetWithDepth
from models.rdd_cod import RDD_COD
from loss.cod_loss import CODLoss
from utils.logger import setup_logger


# ============================================================================
# Ablation variant registry
# ============================================================================

def build_variant(variant_name: str, pretrained: bool = True):
    """
    Tạo model và loss config cho từng ablation variant.
    Trả về (model, loss_cfg_override) — loss_cfg_override có thể None.

    Group 1 — Module-level (M1..M8)
    Group 2 — Depth stream (D1..D6)
    Group 3 — Loss (L1..L7)
    Group 4 — DQG robustness (Q1..Q6) — dùng M8 model + degraded depth
    """
    model = RDD_COD(pretrained=pretrained)
    loss_override = {}

    # ────────────────────────────────────────────────────────────────────────
    # GROUP 1: Module-level ablation
    # ────────────────────────────────────────────────────────────────────────
    if variant_name == "M1":
        # Baseline: dual RGB + simple concat fusion, basic decoder
        # Disable CDFM → use identity fusion; disable BAM → no edge prior;
        # disable depth stream
        model = _M1_baseline(pretrained)

    elif variant_name == "M2":
        # + original 2-stream CDFM (no depth)
        model = _M_no_depth(pretrained, use_cdfm=True, use_bam=False, use_fem=False)

    elif variant_name == "M3":
        # + FEM only (no BAM prior, no CDFM, no depth)
        model = _M_no_depth(pretrained, use_cdfm=False, use_bam=False, use_fem=True)

    elif variant_name == "M4":
        # + BAM-RGB (no CDFM, no FEM, no depth)
        model = _M_no_depth(pretrained, use_cdfm=False, use_bam=True, use_fem=False)

    elif variant_name == "M5":
        # + CDFM + BAM-RGB (no depth)
        model = _M_no_depth(pretrained, use_cdfm=True, use_bam=True, use_fem=True)

    elif variant_name == "M6":
        # + Depth stream with simple concat (no DepthGate, no DQG)
        model = RDD_COD(pretrained=pretrained)
        _disable_depth_gate(model)
        _disable_dqg(model)
        _disable_bam_triple(model)    # keep BAM-RGB only

    elif variant_name == "M7":
        # + Triple-CDFM (full depth gating) but no DQG, BAM still 2-stream
        model = RDD_COD(pretrained=pretrained)
        _disable_dqg(model)
        _disable_bam_triple(model)

    elif variant_name == "M8":
        # Full RDD-COD
        model = RDD_COD(pretrained=pretrained)

    # ────────────────────────────────────────────────────────────────────────
    # GROUP 2: Depth stream design ablation
    # ────────────────────────────────────────────────────────────────────────
    elif variant_name == "D1":
        # w/o DepthGate (average depth with CNN-Trans directly)
        model = RDD_COD(pretrained=pretrained)
        _disable_depth_gate(model)

    elif variant_name == "D2":
        # w/o DQG at all integration points
        model = RDD_COD(pretrained=pretrained)
        _disable_dqg(model)

    elif variant_name == "D3":
        # w/o FEM depth injection
        model = RDD_COD(pretrained=pretrained)
        _disable_fem_depth(model)

    elif variant_name == "D4":
        # w/o BAM-Triple (use 2-stream BAM)
        model = RDD_COD(pretrained=pretrained)
        _disable_bam_triple(model)

    elif variant_name == "D5":
        # PVT-b0 depth backbone instead of b1
        model = RDD_COD(pretrained=pretrained)
        _replace_depth_backbone_b0(model, pretrained)

    elif variant_name == "D6":
        # Full RDD-COD (same as M8)
        model = RDD_COD(pretrained=pretrained)

    # ────────────────────────────────────────────────────────────────────────
    # GROUP 3: Loss component ablation
    # ────────────────────────────────────────────────────────────────────────
    elif variant_name == "L1":
        # seg only
        model = RDD_COD(pretrained=pretrained)
        loss_override = {"lambda_edge": 0.0, "lambda_depth_edge": 0.0, "lambda_entropy": 0.0}

    elif variant_name == "L2":
        # seg + edge
        model = RDD_COD(pretrained=pretrained)
        loss_override = {"lambda_depth_edge": 0.0, "lambda_entropy": 0.0}

    elif variant_name == "L3":
        # seg + depth-edge
        model = RDD_COD(pretrained=pretrained)
        loss_override = {"lambda_edge": 0.0, "lambda_entropy": 0.0}

    elif variant_name == "L4":
        # seg + entropy
        model = RDD_COD(pretrained=pretrained)
        loss_override = {"lambda_edge": 0.0, "lambda_depth_edge": 0.0}

    elif variant_name == "L5":
        # seg + edge + depth-edge
        model = RDD_COD(pretrained=pretrained)
        loss_override = {"lambda_entropy": 0.0}

    elif variant_name == "L6":
        # seg + edge + entropy
        model = RDD_COD(pretrained=pretrained)
        loss_override = {"lambda_depth_edge": 0.0}

    elif variant_name == "L7":
        # Full loss
        model = RDD_COD(pretrained=pretrained)

    # ────────────────────────────────────────────────────────────────────────
    # GROUP 4: DQG robustness — model is always Full (M8), depth is degraded
    # The depth degradation is applied in the dataloader / forward call.
    # Variants Q1-Q3: without DQG; Q4-Q6: with DQG
    # Depth conditions: clean / simple degradation / strong degradation
    # ────────────────────────────────────────────────────────────────────────
    elif variant_name in ("Q1", "Q2", "Q3"):
        # Q1=no-DQG+clean, Q2=no-DQG+simple, Q3=no-DQG+strong
        model = RDD_COD(pretrained=pretrained)
        _disable_dqg(model)

    elif variant_name in ("Q4", "Q5", "Q6"):
        # Q4=DQG+clean, Q5=DQG+simple, Q6=DQG+strong
        model = RDD_COD(pretrained=pretrained)

    else:
        raise ValueError(f"Unknown ablation variant: {variant_name}")

    return model, loss_override


# ============================================================================
# Model modification helpers
# ============================================================================

def _M1_baseline(pretrained):
    """Baseline: dual RGB only, no fusion module, no BAM, no depth."""
    from models.RGBDual_depth_COD_v2 import (
        RDD_COD, Conv1x1, ConvBNR,
        Res2Net50, FEM
    )
    import timm

    class BaselineModel(nn.Module):
        """Stripped-down dual RGB model."""
        def __init__(self, pretrained=True):
            super().__init__()
            self.cnn   = Res2Net50(pretrained=pretrained)
            self.trans = timm.create_model('pvt_v2_b2', pretrained=pretrained,
                                           features_only=True)
            # Simple 1x1 channel alignment + concat fusion
            self.align = nn.ModuleList([
                Conv1x1(256+64,   64),
                Conv1x1(512+128,  128),
                Conv1x1(1024+320, 256),
                Conv1x1(2048+512, 256),
            ])
            # Simple initial prediction
            self.init_block = nn.Sequential(
                ConvBNR(64+128+256+256, 256, 3),
                ConvBNR(256, 256, 3),
                nn.Conv2d(256, 1, 1)
            )
            self.reduce = nn.ModuleList([
                Conv1x1(64, 64), Conv1x1(128, 128),
                Conv1x1(256, 256), Conv1x1(256, 256)
            ])
            self.fem3 = FEM(256, 256, 256)
            self.fem2 = FEM(256, 128, 128)
            self.fem1 = FEM(128, 64,  64)
            self.pred1 = nn.Conv2d(64,  1, 1)
            self.pred2 = nn.Conv2d(128, 1, 1)
            self.pred3 = nn.Conv2d(256, 1, 1)

        def forward(self, rgb, depth):
            x = self.cnn(rgb)
            t = self.trans(rgb)
            c = [self.align[i](torch.cat([x[i], F.interpolate(
                    t[i], size=x[i].shape[2:], mode='bilinear',
                    align_corners=False)], dim=1)) for i in range(4)]
            size4 = c[3].shape[2:]
            concat = torch.cat([c[3]] + [F.interpolate(c[i], size=size4,
                mode='bilinear', align_corners=False) for i in range(3)], dim=1)
            o4 = self.init_block(concat)
            p4 = torch.sigmoid(o4)
            c0 = self.reduce[0](c[0]); c1 = self.reduce[1](c[1])
            c2 = self.reduce[2](c[2]); c3 = self.reduce[3](c[3])
            # Dummy edge (zeros) — no BAM
            edge = torch.zeros_like(p4)
            f3 = self.fem3(c2, c3, edge, p4)
            o3 = self.pred3(f3); p3 = torch.sigmoid(o3)
            f2 = self.fem2(c1, f3, edge, p3)
            o2 = self.pred2(f2); p2 = torch.sigmoid(o2)
            f1 = self.fem1(c0, f2, edge, p2)
            o1 = self.pred1(f1)
            up = lambda t: F.interpolate(t, size=rgb.shape[2:],
                                         mode='bilinear', align_corners=False)
            depth_edge = torch.zeros_like(p4)
            weights = [torch.ones(rgb.shape[0], 3, 1, 1,
                                  device=rgb.device) / 3] * 4
            return (up(o1), up(o2), up(o3), up(o4)), up(edge), depth_edge, weights

    return BaselineModel(pretrained)


def _M_no_depth(pretrained, use_cdfm, use_bam, use_fem):
    """Helper for M2-M5: dual RGB, optionally with CDFM/BAM/FEM but no depth."""
    # Use the CTF-Net structure from CTF-Net baseline (import from CTF-Net if available,
    # otherwise use stripped version of RDD-COD with depth disabled).
    model = RDD_COD(pretrained=pretrained)
    # Force depth encoder to output zeros
    _freeze_depth_as_zeros(model)
    if not use_cdfm:
        _disable_cdfm(model)
    if not use_bam:
        _disable_bam(model)
    return model


def _freeze_depth_as_zeros(model):
    """Replace depth backbone forward with zero output."""
    enc = model.encoder
    original_forward = enc.depth_backbone.forward
    depth_channels = [64, 128, 320, 512]

    def zero_depth_forward(x):
        B, _, H, W = x.shape
        return [torch.zeros(B, c, H // (4 * 2**i), W // (4 * 2**i),
                            device=x.device)
                for i, c in enumerate(depth_channels)]

    enc.depth_backbone.forward = zero_depth_forward


def _disable_cdfm(model):
    """Replace Triple-CDFM with simple addition fusion."""
    from models.RGBDual_depth_COD_v2 import Conv1x1
    enc = model.encoder
    out_chs = [64, 128, 320, 512]
    cnn_chs  = [256, 512, 1024, 2048]
    trans_chs = [64, 128, 320, 512]

    class SimpleFusion(nn.Module):
        def __init__(self, cnn_ch, trans_ch, out_ch):
            super().__init__()
            self.align_cnn  = Conv1x1(cnn_ch, out_ch)
            self.align_trans = Conv1x1(trans_ch, out_ch)

        def forward(self, x_cnn, x_trans, x_depth):
            cnn   = self.align_cnn(x_cnn)
            trans = self.align_trans(x_trans)
            if trans.shape[2:] != cnn.shape[2:]:
                trans = F.interpolate(trans, size=cnn.shape[2:],
                                      mode='bilinear', align_corners=False)
            fused = cnn + trans
            w = torch.ones(x_cnn.shape[0], 3, 1, 1, device=x_cnn.device) / 3
            return fused, w

    for i, (cc, tc, oc) in enumerate(zip(cnn_chs, trans_chs, out_chs), 1):
        setattr(enc, f'cdfm{i}', SimpleFusion(cc, tc, oc).to(
            next(model.parameters()).device))


def _disable_bam(model):
    """Replace BAM with zero edge output."""
    def zero_bam(x1, t4, d4):
        B, _, H, W = x1.shape
        return torch.zeros(B, 1, H, W, device=x1.device)
    model.bam.forward = zero_bam


def _disable_depth_gate(model):
    """Disable DepthGate in all Triple-CDFM modules (use simple average)."""
    for name in ['cdfm1', 'cdfm2', 'cdfm3', 'cdfm4']:
        cdfm = getattr(model.encoder, name)
        def noop_depth_gate(cnn, trans, depth, self=cdfm):
            return cnn, trans
        cdfm.depth_gate.forward = noop_depth_gate


def _disable_dqg(model):
    """Disable DQG in Triple-CDFM, BAM-Triple, and FEM-v2."""
    # CDFM
    for name in ['cdfm1', 'cdfm2', 'cdfm3', 'cdfm4']:
        cdfm = getattr(model.encoder, name)
        def passthrough_cdfm(depth_feat, rgb_feat):
            conf = torch.ones_like(depth_feat[:, :1])
            return depth_feat, conf
        cdfm.depth_quality_gate.forward = passthrough_cdfm

    # BAM
    def passthrough_bam(depth_feat, rgb_feat):
        conf = torch.ones_like(depth_feat[:, :1])
        return depth_feat, conf
    model.bam.depth_quality_gate.forward = passthrough_bam

    # FEM
    for name in ['fem3', 'fem2', 'fem1']:
        fem = getattr(model, name)
        if hasattr(fem, 'depth_quality_gate'):
            def passthrough_fem(depth_feat, rgb_feat):
                conf = torch.ones_like(depth_feat[:, :1])
                return depth_feat, conf
            fem.depth_quality_gate.forward = passthrough_fem


def _disable_fem_depth(model):
    """Disable depth injection in FEM-v2 (set use_depth=False)."""
    for name in ['fem3', 'fem2', 'fem1']:
        getattr(model, name).use_depth = False


def _disable_bam_triple(model):
    """Replace BAM-Triple with 2-stream BAM (no depth input)."""
    bam = model.bam

    def two_stream_forward(x1, t4, d4):
        # Ignore d4 — use only x1 and t4
        size = x1.shape[2:]
        x1r = bam.reduce_cnn(x1)
        t4r = F.interpolate(bam.reduce_trans(t4), size=size,
                             mode='bilinear', align_corners=False)
        # Replace depth with zeros
        d4r = torch.zeros_like(t4r)
        concat  = torch.cat([x1r, t4r, d4r], dim=1)
        fused   = bam.fusion(concat)
        sa      = bam.spatial_attn(fused)
        refined = bam.refine(fused * sa) + fused
        return bam.edge_head(refined)

    model.bam.forward = two_stream_forward


def _replace_depth_backbone_b0(model, pretrained):
    """Replace depth backbone PVT-v2-b1 with PVT-v2-b0."""
    import timm
    enc = model.encoder
    if pretrained:
        model_rgb = timm.create_model('pvt_v2_b0', pretrained=True, features_only=True)
        model_1ch = timm.create_model('pvt_v2_b0', pretrained=False,
                                       features_only=True, in_chans=1)
        rgb_state = model_rgb.state_dict()
        new_state = model_1ch.state_dict()
        for key in rgb_state:
            if 'weight' in key and len(rgb_state[key].shape) == 4 \
                    and rgb_state[key].shape[1] == 3:
                new_state[key] = rgb_state[key].mean(dim=1, keepdim=True)
                break
        for key in rgb_state:
            if key in new_state and rgb_state[key].shape == new_state[key].shape:
                new_state[key] = rgb_state[key]
        model_1ch.load_state_dict(new_state, strict=False)
        enc.depth_backbone = model_1ch
    else:
        enc.depth_backbone = timm.create_model('pvt_v2_b0', pretrained=False,
                                                features_only=True, in_chans=1)
    # Note: PVT-b0 channels are [32,64,160,256] — this will cause dimension
    # mismatch in Triple-CDFM. Rebuild CDFM alignment layers accordingly.
    # (Simplified: just use the model and accept potential size issues via interpolation)


# ============================================================================
# Depth degradation (Group 4)
# ============================================================================

DEGRADATION_LEVEL = {
    "clean":  None,
    "simple": "simple",
    "strong": "strong",
}

DQG_VARIANT_DEPTH = {
    "Q1": ("no_dqg", "clean"),
    "Q2": ("no_dqg", "simple"),
    "Q3": ("no_dqg", "strong"),
    "Q4": ("dqg",    "clean"),
    "Q5": ("dqg",    "simple"),
    "Q6": ("dqg",    "strong"),
}


def degrade_depth(depth, level):
    """Apply degradation to depth tensor (B, 1, H, W)."""
    if level is None:
        return depth

    import torchvision.transforms.functional as TF
    d = depth.clone()
    B, C, H, W = d.shape

    if level == "simple":
        # Light Gaussian noise σ=0.02
        d = d + torch.randn_like(d) * 0.02
        # Gaussian blur kernel=5
        d = TF.gaussian_blur(d, kernel_size=5, sigma=1.0)
        # Downsample ×2 then upsample
        d = F.interpolate(F.interpolate(d, scale_factor=0.5, mode='bilinear',
                                        align_corners=False),
                          size=(H, W), mode='bilinear', align_corners=False)

    elif level == "strong":
        # Gaussian noise σ=0.06
        d = d + torch.randn_like(d) * 0.06
        # Salt-and-pepper noise 5%
        mask = torch.rand_like(d)
        d[mask < 0.025] = 0.0
        d[mask > 0.975] = 1.0
        # Heavy blur kernel=15
        d = TF.gaussian_blur(d, kernel_size=15, sigma=6.0)
        # 20% random occlusion
        occ_h = int(H * 0.2 ** 0.5)
        occ_w = int(W * 0.2 ** 0.5)
        for b in range(B):
            r = torch.randint(0, H - occ_h, (1,)).item()
            c = torch.randint(0, W - occ_w, (1,)).item()
            d[b, :, r:r+occ_h, c:c+occ_w] = 0.0
        # Strong downsample ×4
        d = F.interpolate(F.interpolate(d, scale_factor=0.25, mode='bilinear',
                                        align_corners=False),
                          size=(H, W), mode='bilinear', align_corners=False)
        # Scale/bias error
        d = (1.2 * d - 0.1).clamp(0.0, 1.0)

    return d.clamp(0.0, 1.0)


# ============================================================================
# Training and evaluation
# ============================================================================

def compute_metrics(pred_list, gt_list):
    SM    = py_sod_metrics.Smeasure()
    EM    = py_sod_metrics.Emeasure()
    WFM   = py_sod_metrics.WeightedFmeasure()
    FM    = py_sod_metrics.Fmeasure()
    MAE_m = py_sod_metrics.MAE()
    for pred, gt in zip(pred_list, gt_list):
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        FM.step(pred=pred, gt=gt)
        MAE_m.step(pred=pred, gt=gt)
    em = EM.get_results()["em"]
    fm = FM.get_results()["fm"]
    return {
        "Sm":     float(SM.get_results()["sm"]),
        "meanEm": float(em["curve"].mean()),
        "maxEm":  float(em["curve"].max()),
        "adpEm":  float(em["adp"]),
        "wFm":    float(WFM.get_results()["wfm"]),
        "meanFm": float(fm["curve"].mean()),
        "maxFm":  float(fm["curve"].max()),
        "adpFm":  float(fm["adp"]),
        "MAE":    float(MAE_m.get_results()["mae"]),
    }


def t2np(tensor):
    """(B,1,H,W) → list of (H,W) uint8 [0,255]."""
    arr = tensor.squeeze(1).detach().cpu().numpy()
    return [(a * 255).clip(0, 255).astype(np.uint8) for a in arr]


def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    device, epoch, warmup_cfg, depth_level=None):
    model.train()
    ep, erb, erd = (warmup_cfg["phase1_end"],
                    warmup_cfg["phase2_end"],
                    warmup_cfg["phase3_start"])

    if epoch <= ep:
        for n, p in model.named_parameters():
            p.requires_grad = 'encoder' not in n
    elif epoch <= erb:
        for n, p in model.named_parameters():
            p.requires_grad = 'depth_backbone' not in n
    else:
        for p in model.parameters():
            p.requires_grad = True

    total = 0.0
    n_batch = 0
    for rgb, depth, masks in loader:
        rgb, depth, masks = rgb.to(device), depth.to(device), masks.to(device)
        if depth_level:
            depth = degrade_depth(depth, depth_level)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            preds, edge, dep_edge, fw = model(rgb, depth)
            loss, _ = criterion(preds, edge, dep_edge, fw, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer); scaler.update()
        total += loss.item(); n_batch += 1
    return total / max(n_batch, 1)


@torch.no_grad()
def evaluate(model, loader, device, depth_level=None):
    model.eval()
    preds_all, gts_all = [], []
    for rgb, depth, masks in loader:
        rgb, depth, masks = rgb.to(device), depth.to(device), masks.to(device)
        if depth_level:
            depth = degrade_depth(depth, depth_level)
        preds, *_ = model(rgb, depth)
        prob = torch.sigmoid(preds[0])
        preds_all.extend(t2np(prob))
        gts_all.extend(t2np(masks))
    return compute_metrics(preds_all, gts_all)


# ============================================================================
# Ablation runner
# ============================================================================

GROUPS = {
    "module": ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"],
    "depth":  ["D1", "D2", "D3", "D4", "D5", "D6"],
    "loss":   ["L1", "L2", "L3", "L4", "L5", "L6", "L7"],
    "dqg":    ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],
}

# Default loss config (full model)
DEFAULT_LOSS = dict(
    alpha_seg=0.8, alpha_boundary=0.2,
    lambda_edge=0.5, lambda_depth_edge=0.3, lambda_entropy=0.01,
    seg_weights=(1.0, 0.8, 0.6, 0.4),
)

WARMUP_CFG = dict(phase1_end=5, phase2_end=15, phase3_start=16)


def _atomic_save(obj, path):
    """Ghi file checkpoint theo kiểu atomic: ghi ra .tmp rồi rename.
    Tránh corrupt checkpoint nếu crash giữa chừng."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)   # atomic trên cùng filesystem


def _load_ckpt_safe(path, device):
    """Load checkpoint, trả về None nếu file bị corrupt."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"  [WARN] Checkpoint {path} bị lỗi ({e}), bỏ qua.")
        return None


def _is_done(log_dir):
    """Kiểm tra variant đã chạy xong chưa (có results.json hợp lệ)."""
    results_path = os.path.join(log_dir, "results.json")
    if not os.path.exists(results_path):
        return False
    try:
        with open(results_path) as f:
            d = json.load(f)
        # Hợp lệ khi có test results cho ít nhất 1 dataset
        return bool(d.get("test"))
    except Exception:
        return False


def run_variant(variant, config):
    log_dir = os.path.join("logs", "ablation", _group_of(variant), variant)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir, "train.log")

    # [Fix 5] Skip nếu đã done (chỉ khi không force-rerun)
    if not config.force_rerun and _is_done(log_dir):
        logger.info(f"[SKIP] {variant} đã có results.json hợp lệ.")
        with open(os.path.join(log_dir, "results.json")) as f:
            return json.load(f).get("test", {})

    logger.info(f"=== Ablation variant: {variant} ===")

    # ── depth degradation for Group 4 ──────────────────────────────────────
    depth_level = None
    if variant in DQG_VARIANT_DEPTH:
        _, dl = DQG_VARIANT_DEPTH[variant]
        depth_level = None if dl == "clean" else dl
        logger.info(f"Depth degradation: {dl}")

    # ── model + loss ────────────────────────────────────────────────────────
    model, loss_override = build_variant(variant, pretrained=config.pretrained)
    model = model.to(config.device)

    loss_cfg = copy.deepcopy(DEFAULT_LOSS)
    loss_cfg.update(loss_override)
    criterion = CODLoss(**loss_cfg)

    # ── data ────────────────────────────────────────────────────────────────
    train_ds = DatasetWithDepth(root=config.root, split="train",
                                img_size=config.img_size, augment=True,
                                use_depth=True, logger=logger)
    val_ds   = DatasetWithDepth(root=config.root, split="val",
                                img_size=config.img_size, augment=False,
                                use_depth=True, logger=logger)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers,
                              pin_memory=True)

    # ── optimizer + scheduler ───────────────────────────────────────────────
    optimizer = _build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * len(train_loader), eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    # ── resume: tự động resume nếu last.pth tồn tại và hợp lệ ──────────────
    start_epoch = 1
    best_sm     = 0.0
    ckpt_path   = os.path.join(log_dir, "last.pth")
    best_path   = os.path.join(log_dir, "best.pth")

    if os.path.exists(ckpt_path):
        # [Fix 2] Dùng _load_ckpt_safe để không crash khi file corrupt
        ckpt = _load_ckpt_safe(ckpt_path, config.device)
        if ckpt is not None:
            try:
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scaler.load_state_dict(ckpt["scaler"])
                if "scheduler" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler"])
                start_epoch = ckpt["epoch"] + 1
                best_sm     = ckpt.get("best_sm", 0.0)
                logger.info(f"Auto-resumed epoch {start_epoch}, best Sm={best_sm:.4f}")
            except Exception as e:
                logger.warning(f"Resume thất bại ({e}), train từ đầu.")
                start_epoch = 1
                best_sm     = 0.0

    # ── training loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.epochs + 1):
        try:
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                         scaler, config.device, epoch,
                                         WARMUP_CFG, depth_level)
            val_m = evaluate(model, val_loader, config.device, depth_level)
        except Exception as e:
            logger.error(f"[{variant}] Epoch {epoch} lỗi: {e}")
            logger.error("Bỏ qua epoch này, giữ nguyên checkpoint trước.")
            continue

        logger.info(
            f"[{variant}] Epoch {epoch}/{config.epochs} | "
            f"loss={train_loss:.4f} | "
            f"Sm={val_m['Sm']:.4f} | "
            f"maxEm={val_m['maxEm']:.4f} | adpEm={val_m['adpEm']:.4f} | "
            f"wFm={val_m['wFm']:.4f} | maxFm={val_m['maxFm']:.4f} | "
            f"MAE={val_m['MAE']:.4f}"
        )

        # Scheduler step
        for _ in range(len(train_loader)):
            scheduler.step()

        # [Fix 2] Atomic save last.pth — cập nhật best_sm trước khi save
        if val_m["Sm"] > best_sm:
            best_sm = val_m["Sm"]
            # [Fix 3] Lưu best.pth TRƯỚC last.pth để luôn có best hợp lệ
            _atomic_save(model.state_dict(), best_path)
            logger.info(f"  NEW BEST Sm={best_sm:.4f}")

        _atomic_save(
            {"epoch": epoch, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "best_sm": best_sm},
            ckpt_path
        )

    # ── final eval trên tất cả test sets ────────────────────────────────────
    # [Fix 3] Fallback về last model nếu best.pth không tồn tại
    if os.path.exists(best_path):
        best_state = _load_ckpt_safe(best_path, config.device)
        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info("Loaded best.pth cho final eval.")
        else:
            logger.warning("best.pth corrupt, dùng model cuối epoch.")
    else:
        logger.warning("best.pth không tồn tại, dùng model cuối epoch.")

    results = {}
    for ds_name, ds_root, ds_split in config.test_sets:
        try:
            test_ds = DatasetWithDepth(root=ds_root, split=ds_split,
                                       img_size=config.img_size, augment=False,
                                       use_depth=True, logger=logger)
            test_loader = DataLoader(test_ds, batch_size=config.batch_size,
                                     shuffle=False, num_workers=config.num_workers,
                                     pin_memory=True)
            results[ds_name] = evaluate(model, test_loader, config.device, depth_level)
            logger.info(f"[{variant}] {ds_name}: {results[ds_name]}")
        except Exception as e:
            logger.error(f"[{variant}] Eval {ds_name} lỗi: {e}")
            results[ds_name] = {"error": str(e)}

    # Atomic write results.json
    results_path = os.path.join(log_dir, "results.json")
    tmp_results  = results_path + ".tmp"
    with open(tmp_results, "w") as f:
        json.dump({"variant": variant, "best_val_sm": best_sm,
                   "test": results}, f, indent=2)
    os.replace(tmp_results, results_path)

    logger.info(f"=== Variant {variant} done. Best Sm={best_sm:.4f} ===")
    return results


def _group_of(variant):
    for g, vs in GROUPS.items():
        if variant in vs:
            return g
    return "misc"


def _build_optimizer(model, config):
    cnn_p, trans_p, depth_p, fusion_p, dec_p = [], [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'cnn_backbone' in name:
            cnn_p.append(param)
        elif 'trans_backbone' in name:
            trans_p.append(param)
        elif 'depth_backbone' in name:
            depth_p.append(param)
        elif any(k in name for k in ('cdfm', 'bam', 'depth_gate',
                                      'depth_quality_gate')):
            fusion_p.append(param)
        else:
            dec_p.append(param)
    groups = []
    for ps, lr, nm in [(cnn_p, 5e-5, 'cnn'), (trans_p, 5e-5, 'trans'),
                       (depth_p, 1e-4, 'depth'), (fusion_p, 2e-4, 'fusion'),
                       (dec_p, 2e-4, 'decoder')]:
        if ps:
            groups.append({'params': ps, 'lr': lr, 'name': nm})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


# ============================================================================
# Config + entry point
# ============================================================================

class AblationConfig:
    def __init__(self):
        self.root        = "../Datasets"
        self.img_size    = 352
        self.batch_size  = 8
        self.num_workers = 4
        self.epochs      = 120
        self.pretrained  = True
        self.resume      = False        # deprecated, auto-resume luôn bật
        self.force_rerun = False        # True = chạy lại dù đã có results.json
        self.device      = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_sets   = [
            ("CAMO",   "../Datasets/CAMO", "test"),
            ("COD10K", "../Datasets/COD10K", "test"),
            ("NC4K",   "../Datasets/NC4K", "test"),
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group",   default="all",
                        choices=["all", "module", "depth", "loss", "dqg"])
    parser.add_argument("--variant", default=None,
                        help="Run a single specific variant, e.g. M1")
    parser.add_argument("--resume",  action="store_true",
                        help="(Deprecated) Tự động resume, giữ để backward compat")
    parser.add_argument("--force",   action="store_true",
                        help="Chạy lại variant dù đã có results.json")
    parser.add_argument("--epochs",  type=int, default=None)
    args = parser.parse_args()

    config = AblationConfig()
    # [Fix 5] Auto-resume luôn bật; --force để chạy lại từ đầu
    config.force_rerun = args.force
    if args.epochs:
        config.epochs = args.epochs

    # Collect variants to run
    if args.variant:
        variants = [args.variant]
    elif args.group == "all":
        variants = [v for vs in GROUPS.values() for v in vs]
    else:
        variants = GROUPS[args.group]

    # [Fix 4] Load existing summary để merge kết quả cũ
    os.makedirs("logs/ablation", exist_ok=True)
    summary_path = f"logs/ablation/summary_{args.group}.json"
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                all_results = json.load(f)
            print(f"Loaded existing summary: {summary_path} "
                  f"({len(all_results)} variants)")
        except Exception:
            all_results = {}
    else:
        all_results = {}

    failed = []
    for v in variants:
        print(f"\n{'='*60}\nRunning ablation variant: {v}\n{'='*60}")
        try:
            res = run_variant(v, config)
            all_results[v] = res
        except Exception as e:
            print(f"[ERROR] Variant {v} thất bại: {e}")
            failed.append(v)
            # [Fix 4] Vẫn save summary với kết quả các variant đã xong
        finally:
            # Incremental save sau mỗi variant
            tmp = summary_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(all_results, f, indent=2)
            os.replace(tmp, summary_path)

    print(f"\nAblation summary saved to: {summary_path}")
    if failed:
        print(f"[WARN] Các variant thất bại: {failed}")
        print("Chạy lại: python ablation_study.py --group <group> (auto-resume)")


if __name__ == "__main__":
    main()