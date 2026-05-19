"""
CODLoss — Loss function cho RDD-COD

Kết hợp:
  - BoundaryWeightedLoss: BCE + IoU có boundary weight map
  - BoundaryDiceLoss:     Dice loss cho edge prediction
  - Depth edge supervision: Sobel depth làm pseudo ground-truth cho boundary
  - Entropy regularization: điều tiết CDFM fusion weights

Signature forward():
    loss, loss_dict = criterion(
        predictions, edge_out, depth_edge_map,
        fusion_weights, gt_mask, gt_edge=None,
        override_lambdas=None
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Low-level loss components
# ============================================================================

class BoundaryWeightedLoss(nn.Module):
    """
    Boundary-weighted BCE + IoU loss.

    Weight map: weit = 1 + alpha × |avg_pool(mask) - mask|
    Pixels gần biên nhận weight cao hơn, thúc đẩy model chú ý
    vào vùng ranh giới của đối tượng ngụy trang.

    Args:
        alpha_boundary: Boundary weight multiplier (default: 5.0)
        lambda_bce:     Weight cho BCE component (default: 0.5)
        lambda_iou:     Weight cho IoU component (default: 0.5)
    """
    def __init__(self, alpha_boundary=5.0, lambda_bce=0.5, lambda_iou=0.5):
        super().__init__()
        self.alpha_boundary = alpha_boundary
        total = lambda_bce + lambda_iou
        self.lambda_bce = lambda_bce / total
        self.lambda_iou = lambda_iou / total

    def compute_boundary_weight(self, mask):
        """
        Args:  mask (B, 1, H, W) ∈ [0, 1]
        Returns: weit (B, 1, H, W) ∈ [1, 1 + alpha_boundary]
        """
        blurred  = F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)
        boundary = torch.abs(blurred - mask)
        return 1 + self.alpha_boundary * boundary

    def weighted_bce(self, pred, target, weit):
        """Weighted BCE — pred là logits."""
        bce  = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        wbce = (weit * bce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        return wbce.mean()

    def weighted_iou(self, pred, target, weit):
        """Weighted IoU loss — pred là logits."""
        pred_prob = torch.sigmoid(pred)
        inter = ((pred_prob * target) * weit).sum(dim=(2, 3))
        union = ((pred_prob + target) * weit).sum(dim=(2, 3))
        wiou  = 1 - (inter + 1) / (union - inter + 1)
        return wiou.mean()

    def forward(self, pred, target):
        """
        Args:
            pred:   logits (B, 1, H, W)
            target: ground-truth mask (B, 1, H, W) ∈ [0, 1]
        Returns:
            scalar loss
        """
        weit = self.compute_boundary_weight(target)
        return (self.lambda_bce * self.weighted_bce(pred, target, weit)
                + self.lambda_iou * self.weighted_iou(pred, target, weit))


class BoundaryDiceLoss(nn.Module):
    """
    Dice loss cho boundary/edge prediction.

    Args:
        smooth: Smoothing term (default: 1e-7)
    """
    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred:   sigmoid-activated boundary prediction (B, 1, H, W) ∈ [0, 1]
            target: boundary ground-truth (B, 1, H, W) ∈ [0, 1]
        Returns:
            scalar Dice loss
        """
        pred_flat   = pred.view(-1)
        target_flat = target.view(-1)
        inter = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        return 1 - (2. * inter + self.smooth) / (union + self.smooth)


# ============================================================================
# CODLoss — main loss class
# ============================================================================

class CODLoss(nn.Module):
    """
    CODLoss cho RDD-COD.

    Bốn thành phần loss:
      1. Segmentation loss  — BoundaryWeightedLoss tại 4 scales (deep supervision).
                              Weights được normalize nội bộ: Σw_i = 1.
      2. Boundary loss      — BoundaryDiceLoss(edge_pred, gt_edge) nếu gt_edge được cung cấp.
      3. Depth edge loss    — BoundaryDiceLoss(edge_pred, Sobel(depth)) dùng pseudo GT từ depth.
      4. Entropy reg        — Maximize entropy của CDFM fusion weights để tránh
                              một nhánh bị dominated.

    Tổng loss:
        L = α_seg × L_seg
          + λ_edge       × L_boundary      (nếu gt_edge không None)
          + λ_depth_edge × L_depth_edge
          + λ_entropy    × L_entropy

    Để tắt boundary-related losses trong giai đoạn warmup:
        loss, d = criterion(..., override_lambdas={'lambda_edge': 0.0,
                                                    'lambda_depth_edge': 0.0})

    Args:
        alpha_seg:          Weight tổng cho segmentation (default: 0.8)
        alpha_boundary:     Weight tổng cho boundary — dùng để tính λ_edge
                            khi lambda_edge không được truyền trực tiếp.
                            (normalize: λ_edge = alpha_boundary / (alpha_seg + alpha_boundary))
        ds_weights:         Deep supervision weights [d1, d2, d3, d4] — normalize nội bộ
        seg_weights:        Alias của ds_weights
        lambda_edge:        Ghi đè trực tiếp λ_edge, bỏ qua alpha_boundary nếu được set
        boundary_emphasis:  alpha_boundary trong BoundaryWeightedLoss (default: 5.0)
        lambda_depth_edge:  Weight cho depth edge supervision (default: 0.3)
        lambda_entropy:     Weight cho entropy regularization (default: 0.01)
        depth_edge_thresh:  Ngưỡng binarize Sobel depth edge map (default: 0.1)
    """
    def __init__(
        self,
        alpha_seg         = 0.8,
        alpha_boundary    = 0.2,
        ds_weights        = None,
        seg_weights       = None,
        lambda_edge       = None,
        boundary_emphasis = 5.0,
        lambda_depth_edge = 0.3,
        lambda_entropy    = 0.01,
        depth_edge_thresh = 0.1,
    ):
        super().__init__()

        # Nếu lambda_edge được truyền trực tiếp thì dùng luôn,
        # không thì normalize từ alpha_seg và alpha_boundary
        if lambda_edge is not None:
            self.alpha_seg   = alpha_seg
            self.lambda_edge = lambda_edge
        else:
            total = alpha_seg + alpha_boundary
            self.alpha_seg   = alpha_seg      / total
            self.lambda_edge = alpha_boundary / total

        # Deep supervision weights — normalize nội bộ
        raw_weights = seg_weights if seg_weights is not None else ds_weights
        if raw_weights is None:
            raw_weights = [1.0, 0.5, 0.3, 0.2]
        ds_sum = sum(raw_weights)
        self.ds_weights = [w / ds_sum for w in raw_weights]

        self.lambda_depth_e    = lambda_depth_edge
        self.lambda_entropy    = lambda_entropy
        self.depth_edge_thresh = depth_edge_thresh

        # Loss components
        self.seg_loss      = BoundaryWeightedLoss(alpha_boundary=boundary_emphasis)
        self.boundary_loss = BoundaryDiceLoss()


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def entropy_reg(weights_list):
        """
        Entropy regularization trên CDFM fusion weights.

        Maximize entropy để tránh một nhánh (global/local/depth) bị dominated,
        khuyến khích model sử dụng cả ba nhánh một cách cân bằng.

        Args:
            weights_list: list of (B, 3, 1, 1) tensors từ TripleCDFM
        Returns:
            scalar loss (minimize → maximize entropy)
        """
        loss = 0.0
        for w in weights_list:
            w = w.squeeze(-1).squeeze(-1)                   # (B, 3)
            entropy = -(w * (w + 1e-8).log()).sum(dim=1).mean()
            loss   += -entropy                              # minimize -entropy
        return loss / len(weights_list)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        predictions,
        edge_pred,
        depth_edge_map,
        fusion_weights,
        gt_mask,
        gt_edge          = None,
        override_lambdas = None,
    ):
        """
        Args:
            predictions:      tuple (pred_d1, pred_d2, pred_d3, pred_d4) — logits, đã upsample
            edge_pred:        (B, 1, H, W) sigmoid activated edge prediction
            depth_edge_map:   (B, 1, H, W) từ compute_depth_edge() trong model
            fusion_weights:   list of (B, 3, 1, 1) từ TripleCDFM
            gt_mask:          (B, 1, H, W) ground-truth segmentation ∈ [0, 1]
            gt_edge:          (B, 1, H, W) ground-truth edge map ∈ [0, 1] (optional)
            override_lambdas: dict ghi đè lambda trong warmup,
                              e.g. {'lambda_edge': 0.0, 'lambda_depth_edge': 0.0}

        Returns:
            total_loss: scalar tensor
            loss_dict:  dict breakdown các thành phần để log
        """
        # Resolve lambdas — có thể bị override trong warmup
        lambda_edge    = self.lambda_edge
        lambda_depth_e = self.lambda_depth_e
        if override_lambdas:
            lambda_edge    = override_lambdas.get('lambda_edge',       lambda_edge)
            lambda_depth_e = override_lambdas.get('lambda_depth_edge', lambda_depth_e)

        H, W = gt_mask.shape[2:]

        # ── 1. Segmentation loss (deep supervision) ───────────────────────
        seg_loss = 0.0
        for pred, w in zip(predictions, self.ds_weights):
            if pred.shape[2:] != (H, W):
                pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
            seg_loss = seg_loss + w * self.seg_loss(pred, gt_mask)
        seg_loss = self.alpha_seg * seg_loss

        # ── 2. Boundary loss (gt_edge nếu có) ────────────────────────────
        edge_loss = torch.tensor(0.0, device=gt_mask.device)
        if gt_edge is not None and lambda_edge > 0:
            edge_pred_up = edge_pred
            if edge_pred.shape[2:] != (H, W):
                edge_pred_up = F.interpolate(edge_pred, size=(H, W),
                                             mode='bilinear', align_corners=False)
            edge_loss = self.boundary_loss(edge_pred_up, gt_edge.clamp(0, 1))

        # ── 3. Depth edge supervision ─────────────────────────────────────
        depth_e_loss = torch.tensor(0.0, device=gt_mask.device)
        if lambda_depth_e > 0:
            depth_edge_up = F.interpolate(depth_edge_map, size=(H, W),
                                          mode='bilinear', align_corners=False)
            depth_edge_gt = (depth_edge_up > self.depth_edge_thresh).float()
            edge_pred_up  = edge_pred
            if edge_pred.shape[2:] != (H, W):
                edge_pred_up = F.interpolate(edge_pred, size=(H, W),
                                             mode='bilinear', align_corners=False)
            depth_e_loss = self.boundary_loss(edge_pred_up, depth_edge_gt)

        # ── 4. Entropy regularization ─────────────────────────────────────
        ent_loss = self.entropy_reg(fusion_weights)

        # ── Total ─────────────────────────────────────────────────────────
        total = (seg_loss
                 + lambda_edge    * edge_loss
                 + lambda_depth_e * depth_e_loss
                 + self.lambda_entropy * ent_loss)

        loss_dict = {
            'seg_loss':        seg_loss.item(),
            'edge_loss':       edge_loss.item(),
            'depth_edge_loss': depth_e_loss.item(),
            'entropy_loss':    ent_loss.item(),
            'total':           total.item(),
        }
        return total, loss_dict


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    loss_fn = CODLoss(
        alpha_seg         = 0.8,
        alpha_boundary    = 0.2,
        ds_weights        = [1.0, 0.5, 0.3, 0.2],
        boundary_emphasis = 5.0,
        lambda_depth_edge = 0.3,
        lambda_entropy    = 0.01,
    )

    print("Normalized weights:")
    print(f"  alpha_seg   : {loss_fn.alpha_seg:.4f}")
    print(f"  lambda_edge : {loss_fn.lambda_edge:.4f}")
    print(f"  ds_weights  : {[f'{w:.4f}' for w in loss_fn.ds_weights]}")
    print(f"  sum(ds_w)   : {sum(loss_fn.ds_weights):.4f}")

    B, H, W = 2, 352, 352
    predictions    = (torch.randn(B, 1, H, W),
                      torch.randn(B, 1, H//2, W//2),
                      torch.randn(B, 1, H//4, W//4),
                      torch.randn(B, 1, H//8, W//8))
    edge_pred      = torch.sigmoid(torch.randn(B, 1, H, W))
    depth_edge_map = torch.rand(B, 1, H, W)
    fusion_weights = [torch.softmax(torch.randn(B, 3, 1, 1), dim=1) for _ in range(4)]
    gt_mask        = torch.randint(0, 2, (B, 1, H, W)).float()
    gt_edge        = torch.randint(0, 2, (B, 1, H, W)).float()

    # Normal forward
    loss, d = loss_fn(predictions, edge_pred, depth_edge_map,
                      fusion_weights, gt_mask, gt_edge=gt_edge)
    print(f"\nTotal loss: {loss.item():.4f}")
    print("Breakdown:")
    for k, v in d.items():
        print(f"  {k}: {v:.4f}")

    # Warmup override
    loss_w, d_w = loss_fn(predictions, edge_pred, depth_edge_map,
                           fusion_weights, gt_mask, gt_edge=gt_edge,
                           override_lambdas={'lambda_edge': 0.0,
                                             'lambda_depth_edge': 0.0})
    print(f"\nWarmup loss (boundary disabled): {loss_w.item():.4f}")
    print("Breakdown:")
    for k, v in d_w.items():
        print(f"  {k}: {v:.4f}")