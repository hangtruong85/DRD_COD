"""
Evaluation script for DRD-COD
Triple-stream architecture: RGB-CNN (Res2Net-50) + RGB-Transformer (PVT-v2-b2) + Depth (PVT-v2-b1)
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

import py_sod_metrics

from datasets.mhcd_dataset import DatasetWithDepth
# from datasets.cod10k_dataset import DatasetWithDepth
# from datasets.camo_dataset import DatasetWithDepth
# from datasets.camo_cod10k_dataset import DatasetWithDepth
# from datasets.nc4k_dataset import DatasetWithDepth

from models.drd_cod import DRD_COD
from utils.logger import setup_logger


# =========================================================
# Configuration
# =========================================================
class EvalConfig:
    def __init__(self):
        self.n_classes  = 1
        self.pretrained = False   # weights loaded from checkpoint

        self.ckpt_path = "logs/DRD_COD_20260421_100747/best.pth"

        # Dataset
        self.root = "../Datasets/COD10K"
        #self.root   = "../Datasets/CAMO"
        #self.root = "../Datasets/NC4K"
        self.split       = "test"
        self.img_size    = 352
        self.batch_size  = 8
        self.num_workers = 4
        self.use_depth   = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Output
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = f"eval_results/DRD_COD_{self.split}_{ts}"
        os.makedirs(self.save_dir, exist_ok=True)

        self.save_predictions = False
        self.num_visualize    = 10


# =========================================================
# Helpers
# =========================================================
def tensor_to_numpy_uint8(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(1)
    elif tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().numpy()
    return (arr * 255).clip(0, 255).astype(np.uint8)


def load_checkpoint(model, ckpt_path, device, logger):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        logger.info(f"  Loaded from epoch {ckpt.get('epoch', '?')}, "
                    f"best Sm: {ckpt.get('best_s_measure', '?')}")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
        logger.info("  Loaded state_dict")
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
        logger.info("  Loaded state_dict directly")
    else:
        raise RuntimeError("Invalid checkpoint format")

    return model


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate_model(model, dataloader, device, config, logger):
    model.eval()

    SM    = py_sod_metrics.Smeasure()
    EM    = py_sod_metrics.Emeasure()
    WFM   = py_sod_metrics.WeightedFmeasure()
    FM    = py_sod_metrics.Fmeasure()
    MAE_m = py_sod_metrics.MAE()

    logger.info("Starting evaluation...")
    logger.info("=" * 60)
    logger.info("Model: DRD_COD")
    logger.info("  - RGB-CNN:   Res2Net-50")
    logger.info("  - RGB-Trans: PVT-v2-b2")
    logger.info("  - Depth:     PVT-v2-b1")
    logger.info("  - Fusion:    DCFM")
    logger.info("  - Boundary:  TBAM")
    logger.info("  - Decoder:   GRD")
    logger.info("=" * 60)
    logger.info("Using pred_d1 (finest scale) for metrics")

    num_samples = 0
    predictions_to_save = []

    pbar = tqdm(dataloader, desc="Evaluating", ncols=100)

    for rgb, depth, masks in pbar:
        rgb   = rgb.to(device)
        depth = depth.to(device)
        masks = masks.to(device)

        predictions, edge_out, _depth_edge_map, _fusion_weights = model(rgb, depth)
        pred_d1 = predictions[0]   # finest scale

        pred_probs = torch.sigmoid(pred_d1)

        pred_np = tensor_to_numpy_uint8(pred_probs)
        gt_np   = tensor_to_numpy_uint8(masks)

        batch_size = rgb.shape[0]
        for i in range(batch_size):
            SM.step(pred=pred_np[i], gt=gt_np[i])
            EM.step(pred=pred_np[i], gt=gt_np[i])
            WFM.step(pred=pred_np[i], gt=gt_np[i])
            FM.step(pred=pred_np[i], gt=gt_np[i])
            MAE_m.step(pred=pred_np[i], gt=gt_np[i])

            if config.save_predictions and num_samples + i < config.num_visualize:
                predictions_to_save.append({
                    'pred': pred_np[i], 'gt': gt_np[i], 'idx': num_samples + i
                })

        num_samples += batch_size

        if num_samples >= 10:
            pbar.set_postfix({
                'Sm':  f"{SM.get_results()['sm']:.4f}",
                'MAE': f"{MAE_m.get_results()['mae']:.4f}",
                'n':   num_samples
            })

    sm  = SM.get_results()["sm"]
    em  = EM.get_results()["em"]
    wfm = WFM.get_results()["wfm"]
    fm  = FM.get_results()["fm"]
    mae = MAE_m.get_results()["mae"]

    results = {
        "Sm":      float(sm),
        "meanEm":  float(em["curve"].mean()),
        "maxEm":   float(em["curve"].max()),
        "adpEm":   float(em["adp"]),
        "wFm":     float(wfm),
        "meanFm":  float(fm["curve"].mean()),
        "maxFm":   float(fm["curve"].max()),
        "adpFm":   float(fm["adp"]),
        "MAE":     float(mae),
        "num_samples": num_samples,
    }

    if config.save_predictions and predictions_to_save:
        save_visualizations(predictions_to_save, config.save_dir)
        logger.info(f"Saved {len(predictions_to_save)} prediction visualizations")

    return results


def save_visualizations(predictions, save_dir):
    import cv2
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    for item in predictions:
        idx = item['idx']
        cv2.imwrite(os.path.join(vis_dir, f"{idx:04d}_pred.png"),    item['pred'])
        cv2.imwrite(os.path.join(vis_dir, f"{idx:04d}_gt.png"),      item['gt'])
        cv2.imwrite(os.path.join(vis_dir, f"{idx:04d}_compare.png"),
                    np.hstack([item['gt'], item['pred']]))


# =========================================================
# Display & Save Results
# =========================================================
def display_results(results, logger):
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS — DRD-COD")
    print("=" * 80)

    metrics_display = [
        ("Sm",     results["Sm"],     "Structure Measure"),
        ("meanEm", results["meanEm"], "Mean E-measure"),
        ("maxEm",  results["maxEm"],  "Max E-measure"),
        ("adpEm",  results["adpEm"],  "Adaptive E-measure"),
        ("wFm",    results["wFm"],    "Weighted F-measure"),
        ("meanFm", results["meanFm"], "Mean F-measure"),
        ("maxFm",  results["maxFm"],  "Max F-measure"),
        ("adpFm",  results["adpFm"],  "Adaptive F-measure"),
        ("MAE",    results["MAE"],    "Mean Absolute Error"),
    ]

    print(f"\n  {'Metric':<10} {'Value':>10}   {'Description':<25}")
    print("  " + "-" * 50)
    for name, val, desc in metrics_display:
        star = " ★" if name in ("Sm", "maxEm", "wFm", "MAE") else ""
        print(f"  {name:<10} {val:>10.4f}   {desc:<25}{star}")
    print("  " + "-" * 50)
    print(f"  Total Samples: {results['num_samples']}")
    print("=" * 80 + "\n")

    logger.info("EVALUATION RESULTS")
    logger.info("-" * 40)
    for name, val, desc in metrics_display:
        logger.info(f"  {name:<10}: {val:.4f}")
    logger.info(f"  Samples: {results['num_samples']}")


def save_results(results, config, logger):
    summary = {
        "model": "DRD_COD",
        "architecture": {
            "rgb_cnn":   "Res2Net-50",
            "rgb_trans": "PVT-v2-b2",
            "depth":     "PVT-v2-b1",
            "fusion":    "CDFM",
            "boundary":  "TBAM",
            "decoder":   "GRD",
        },
        "metrics": results,
        "config": {
            "checkpoint":   config.ckpt_path,
            "dataset_root": config.root,
            "split":        config.split,
            "img_size":     config.img_size,
            "use_depth":    config.use_depth,
        },
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_path = os.path.join(config.save_dir, "results.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Results saved to: {summary_path}")

    csv_path = os.path.join(config.save_dir, "metrics.csv")
    with open(csv_path, 'w') as f:
        f.write("Metric,Value\n")
        for key, val in results.items():
            f.write(f"{key},{val:.6f}\n" if isinstance(val, float) else f"{key},{val}\n")
    logger.info(f"Metrics CSV saved to: {csv_path}")

    txt_path = os.path.join(config.save_dir, "summary.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("DRD-COD Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint : {config.ckpt_path}\n")
        f.write(f"Dataset    : {config.root} / {config.split}\n")
        f.write(f"Samples    : {results['num_samples']}\n\n")
        f.write("-" * 30 + "\n")
        f.write("KEY METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"  Sm   : {results['Sm']:.4f}\n")
        f.write(f"  maxEm: {results['maxEm']:.4f}\n")
        f.write(f"  wFm  : {results['wFm']:.4f}\n")
        f.write(f"  MAE  : {results['MAE']:.4f}\n")
        f.write("-" * 30 + "\n\n")
        f.write("ALL METRICS:\n")
        for key, val in results.items():
            if isinstance(val, float):
                f.write(f"  {key:<10}: {val:.4f}\n")
    logger.info(f"Summary saved to: {txt_path}")


# =========================================================
# Main
# =========================================================
def main():
    config = EvalConfig()
    logger = setup_logger(config.save_dir, "evaluation.log")

    logger.info("=" * 80)
    logger.info("DRD-COD EVALUATION")
    logger.info("=" * 80)
    logger.info("  RGB-CNN:   Res2Net-50")
    logger.info("  RGB-Trans: PVT-v2-b2")
    logger.info("  Depth:     PVT-v2-b1")
    logger.info("  Fusion:    DCFM")
    logger.info("  Boundary:  TBAM")
    logger.info("  Decoder:   GRD")
    logger.info("")
    logger.info(f"Checkpoint : {config.ckpt_path}")
    logger.info(f"Dataset    : {config.root} / {config.split}")
    logger.info(f"Image size : {config.img_size}")
    logger.info(f"Device     : {config.device}")
    logger.info("=" * 80)

    logger.info("Creating model...")
    model = DRD_COD(
        n_classes=config.n_classes, pretrained=config.pretrained
    ).to(config.device)

    total_p, _ = count_parameters(model)
    logger.info(f"Parameters: {total_p:,}")

    model = load_checkpoint(model, config.ckpt_path, config.device, logger)

    logger.info(f"Loading dataset: {config.split}")
    dataset = DatasetWithDepth(
        root=config.root, split=config.split, img_size=config.img_size,
        augment=False, use_depth=config.use_depth, logger=logger
    )
    logger.info(f"Total samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )

    results = evaluate_model(model, dataloader, config.device, config, logger)

    display_results(results, logger)
    save_results(results, config, logger)

    print("\n" + "=" * 50)
    print("QUICK SUMMARY")
    print("=" * 50)
    print(f"  Sm    : {results['Sm']:.4f}")
    print(f"  meanEm: {results['meanEm']:.4f}")
    print(f"  wFm   : {results['wFm']:.4f}")
    print(f"  MAE   : {results['MAE']:.4f}")
    print("=" * 50)
    print(f"\nResults saved to: {config.save_dir}")


if __name__ == "__main__":
    main()