"""
RDD-COD
Triple-stream architecture: RGB-CNN (Res2Net) + RGB-Transformer (PVT) + Depth (PVT)

Architecture:
    RGB ──┬──► Res2Net-50   ──► x1,x2,x3,x4  (CNN features)
          │
          └──► PVT-v2-b2    ──► t1,t2,t3,t4  (Transformer features)

    Depth ──► PVT-v2-b1     ──► d1,d2,d3,d4  (Depth features)

    TripleCDFM at each stage:
        depth gate → CNN enhanced
        depth gate → Trans enhanced
        Adaptive fusion (global + local + depth branches)

    BAM-Triple: Edge detection using x1 + t4 + d4

    FEM Decoder: Progressive refinement with edge guidance + depth features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from math import log


# ============================================================================
# Res2Net-50 Backbone
# ============================================================================
class Res2Net50(nn.Module):
    """
    Res2Net-50 backbone for local feature extraction.
    Output channels: [256, 512, 1024, 2048] at scales [1/4, 1/8, 1/16, 1/32]
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'res2net50_26w_4s',
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        self.channels = [256, 512, 1024, 2048]

    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# Basic Building Blocks
# ============================================================================
class ConvBNR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out    = torch.mean(x, dim=1, keepdim=True)
        attn       = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        return attn


class ChannelAttention(nn.Module):
    """ECA-style channel attention."""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        t = int(abs((log(channels, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv1d  = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        combined = self.avg_pool(x) + self.max_pool(x)
        attn = self.conv1d(combined.squeeze(-1).transpose(-1, -2))
        attn = attn.transpose(-1, -2).unsqueeze(-1)
        return self.sigmoid(attn)


# ============================================================================
# DepthGate: depth làm cross-attention gate cho CNN và Trans
# ============================================================================
class DepthGate(nn.Module):
    """
    Depth-guided cross-attention gate.

    Depth channel attention được dùng để gate CNN features,
    depth spatial attention được dùng để gate Transformer features.
    Depth chủ động điều chỉnh cả hai stream riêng biệt thay vì
    tác động lên đặc trưng trung bình của hai stream.
    """
    def __init__(self, channels):
        super().__init__()
        # Channel attention (ECA) từ depth để gate CNN
        t = int(abs((log(channels, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.depth_ca_pool  = nn.AdaptiveAvgPool2d(1)
        self.depth_ca_conv  = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.depth_ca_sig   = nn.Sigmoid()

        # Spatial attention từ depth để gate Trans
        self.depth_sa_conv  = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.depth_sa_sig   = nn.Sigmoid()

    def forward(self, cnn, trans, depth):
        """
        Args:
            cnn, trans, depth: tất cả đã được align về cùng (B, C, H, W)
        Returns:
            cnn_gated, trans_gated
        """
        # Depth channel attention → gate CNN
        ca = self.depth_ca_pool(depth)
        ca = self.depth_ca_conv(ca.squeeze(-1).transpose(-1, -2))
        ca = self.depth_ca_sig(ca.transpose(-1, -2).unsqueeze(-1))
        cnn_gated = cnn * ca

        # Depth spatial attention → gate Trans
        d_max, _ = torch.max(depth, dim=1, keepdim=True)
        d_avg    = torch.mean(depth, dim=1, keepdim=True)
        sa       = self.depth_sa_sig(self.depth_sa_conv(torch.cat([d_max, d_avg], dim=1)))
        trans_gated = trans * sa

        return cnn_gated, trans_gated


# ============================================================================
# DepthQualityGate: học confidence score cho depth stream
# ============================================================================
class DepthQualityGate(nn.Module):
    """
    Confidence-based depth quality gate.

    So sánh depth feature với RGB feature để tính cosine similarity:
    độ tương đồng cao cho thấy depth nhất quán với RGB, dẫn đến
    confidence cao. Global pooling gate và local disagreement map
    được kết hợp để tạo ra scalar confidence ∈ (0, 1) tại mỗi vị trí.

    Khi confidence cao, depth feature được dùng đầy đủ.
    Khi confidence thấp (depth nhiễu), module fallback về RGB feature.
    Confidence được học tự động từ reconstruction loss qua backprop.
    """
    def __init__(self, channels):
        super().__init__()
        # Global gate từ depth statistics
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1, bias=False),
            nn.Sigmoid()
        )
        # Local gate từ depth-RGB disagreement
        self.local_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1, bias=False),
            nn.Sigmoid()
        )
        # Blend global + local confidence
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, depth_feat, rgb_feat):
        """
        Args:
            depth_feat: (B, C, H, W)
            rgb_feat:   (B, C, H, W) — CNN hoặc Trans feature cùng scale
        Returns:
            gated:      (B, C, H, W) — depth contribution đã được quality-filtered
            confidence: (B, 1, H, W)
        """
        # Global confidence: từ depth feature pool
        g_conf = self.global_gate(depth_feat)                               # (B, 1, 1, 1)

        # Local confidence: từ depth-RGB disagreement map
        l_conf = self.local_gate(torch.cat([depth_feat, rgb_feat], dim=1))  # (B, 1, H, W)

        # Blend: alpha học được quyết định tỷ lệ global vs local
        alpha  = torch.sigmoid(self.alpha)
        conf   = alpha * g_conf + (1 - alpha) * l_conf                      # (B, 1, H, W)

        # Gate: confidence cao → dùng depth, thấp → fallback RGB
        gated  = conf * depth_feat + (1 - conf) * rgb_feat

        return gated, conf


# ============================================================================
# BAM-Triple: Boundary Attention Module với 3 streams
# ============================================================================
class BAMTriple(nn.Module):
    """
    Boundary Attention Module sử dụng ba nguồn đặc trưng:
      x1 — CNN low-level features (chi tiết cạnh)
      t4 — Transformer high-level features (ngữ cảnh ngữ nghĩa)
      d4 — Depth high-level features (biên giới hình học)

    Depth contribution được lọc qua DepthQualityGate trước khi concat
    để hạn chế ảnh hưởng của depth nhiễu lên boundary prediction.
    """
    def __init__(self, cnn_channels=256, trans_channels=512, depth_channels=512):
        super().__init__()
        hidden = 256
        self.reduce_cnn   = Conv1x1(cnn_channels, 64)
        self.reduce_trans = Conv1x1(trans_channels, hidden)
        self.reduce_depth = Conv1x1(depth_channels, hidden)

        self.fusion = nn.Sequential(
            ConvBNR(hidden + hidden + 64, hidden, 3),
            ConvBNR(hidden, hidden, 3)
        )
        self.spatial_attn = SpatialAttention(kernel_size=7)
        self.refine   = ConvBNR(hidden, hidden, 3)
        self.edge_head = nn.Conv2d(hidden, 1, 1)

        self.depth_quality_gate = DepthQualityGate(hidden)

    def forward(self, x1, t4, d4):
        size = x1.shape[2:]
        x1r  = self.reduce_cnn(x1)
        t4r  = F.interpolate(self.reduce_trans(t4), size=size, mode='bilinear', align_corners=False)
        d4r  = F.interpolate(self.reduce_depth(d4), size=size, mode='bilinear', align_corners=False)

        # Gate depth contribution — dùng t4r làm RGB reference
        d4r, _ = self.depth_quality_gate(d4r, t4r)

        concat  = torch.cat([x1r, t4r, d4r], dim=1)
        fused   = self.fusion(concat)
        sa      = self.spatial_attn(fused)
        refined = self.refine(fused * sa) + fused
        return self.edge_head(refined)


# ============================================================================
# TripleCDFM: Triple CNN-Depth-Transformer Fusion Module
# ============================================================================
class TripleCDFM(nn.Module):
    """
    Triple CNN-Depth-Transformer Fusion Module.

    Depth gate điều chỉnh CNN và Trans riêng biệt:
      - DepthGate dùng channel attention của depth để gate CNN
      - DepthGate dùng spatial attention của depth để gate Trans

    Ba nhánh fusion song song:
      - Global branch: Trans-gated → channel attention → gate CNN
      - Local branch:  CNN-gated  → spatial attention  → gate Trans
      - Depth branch:  depth-gated trung bình CNN+Trans được lọc qua DepthQualityGate

    Adaptive weights (Softmax) học để blend ba nhánh theo từng input.
    """
    def __init__(self, cnn_channels, trans_channels, depth_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels

        # Channel alignment
        self.align_cnn   = Conv1x1(cnn_channels, out_channels)
        self.align_trans = Conv1x1(trans_channels, out_channels)
        self.align_depth = Conv1x1(depth_channels, out_channels)

        self.depth_gate = DepthGate(out_channels)

        # Global branch (Trans → CNN)
        self.global_ca   = ChannelAttention(out_channels)
        self.global_conv = Conv1x1(out_channels * 2, out_channels)

        # Local branch (CNN → Trans)
        self.local_sa = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        self.local_conv = Conv1x1(out_channels * 2, out_channels)

        # Depth branch
        self.depth_refine = ConvBNR(out_channels * 2, out_channels, 3)

        # DepthQualityGate — filter noisy depth trước weighted fusion
        self.depth_quality_gate = DepthQualityGate(out_channels)

        # Adaptive fusion weights (3 branches)
        self.fusion_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 3, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 3, 1),
            nn.Softmax(dim=1)
        )

        self.final_fusion = ConvBNR(out_channels, out_channels, 3)

    def forward(self, x_cnn, x_trans, x_depth):
        size = x_cnn.shape[2:]

        # Align channels
        cnn   = self.align_cnn(x_cnn)
        trans = self.align_trans(x_trans)
        depth = self.align_depth(x_depth)

        # Spatial alignment
        if trans.shape[2:] != size:
            trans = F.interpolate(trans, size=size, mode='bilinear', align_corners=False)
        if depth.shape[2:] != size:
            depth = F.interpolate(depth, size=size, mode='bilinear', align_corners=False)

        # Depth-guided gating cho CNN và Trans
        cnn_gated, trans_gated = self.depth_gate(cnn, trans, depth)

        # Global branch: Trans-gated → gate CNN với channel attention
        ca_weight  = self.global_ca(trans_gated)
        global_out = self.global_conv(torch.cat([cnn_gated * ca_weight, trans_gated], dim=1))

        # Local branch: CNN-gated → gate Trans với spatial attention
        sa_weight = self.local_sa(cnn_gated)
        local_out = self.local_conv(torch.cat([trans_gated * sa_weight, cnn_gated], dim=1))

        # Depth branch: depth-gated cnn+trans fusion
        depth_out = self.depth_refine(
            torch.cat([(cnn_gated + trans_gated) / 2, depth], dim=1)
        )

        # DepthQualityGate: filter depth_out trước khi fuse
        # rgb_ref = average của global + local làm fallback reference
        rgb_ref   = (global_out + local_out) / 2
        depth_out, depth_conf = self.depth_quality_gate(depth_out, rgb_ref)

        # Adaptive weighted fusion
        concat  = torch.cat([global_out, local_out, depth_out], dim=1)
        weights = self.fusion_weights(concat)   # (B, 3, 1, 1)

        fused = (weights[:, 0:1] * global_out
                 + weights[:, 1:2] * local_out
                 + weights[:, 2:3] * depth_out)
        fused = self.final_fusion(fused)

        return fused, weights   # trả weights để tính entropy loss nếu cần


# ============================================================================
# FEM: Feature Enhancement Module
# ============================================================================
class FEM(nn.Module):
    """
    Feature Enhancement Module.

    Kết hợp high-level và low-level features có hướng dẫn từ edge attention
    và prediction từ stage trước. Depth features được dùng để tạo thêm
    một attention map bổ sung, được lọc qua DepthQualityGate để loại bỏ
    ảnh hưởng của depth nhiễu.

    Adaptive weights (sigmoid) học tỷ lệ đóng góp giữa high-level và
    low-level features tại mỗi vị trí không gian.
    """
    def __init__(self, high_channels, low_channels, out_channels, depth_channels=None):
        super().__init__()
        self.align_high = Conv1x1(high_channels, out_channels)
        self.align_low  = Conv1x1(low_channels, out_channels)

        if depth_channels is not None:
            self.use_depth  = True
            self.depth_gate = nn.Sequential(
                Conv1x1(depth_channels, out_channels // 4),
                nn.Conv2d(out_channels // 4, 1, 1),
                nn.Sigmoid()
            )
            self.depth_quality_gate = DepthQualityGate(out_channels // 4)
        else:
            self.use_depth = False

        self.weight_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, 2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)
        self.output_conv = ConvBNR(out_channels * 2, out_channels, 3)

    def forward(self, low_feat, high_feat, edge_attn, prev_pred, depth_feat=None):
        size = low_feat.shape[2:]

        high_feat = self.align_high(high_feat)
        if high_feat.shape[2:] != size:
            high_feat = F.interpolate(high_feat, size=size, mode='bilinear', align_corners=False)

        low_feat = self.align_low(low_feat)

        if edge_attn.shape[2:] != size:
            edge_attn = F.interpolate(edge_attn, size=size, mode='bilinear', align_corners=False)
        if prev_pred.shape[2:] != size:
            prev_pred = F.interpolate(prev_pred, size=size, mode='bilinear', align_corners=False)

        attn = edge_attn + prev_pred

        # Depth attention với quality gate
        if self.use_depth and depth_feat is not None:
            if depth_feat.shape[2:] != size:
                depth_feat = F.interpolate(depth_feat, size=size, mode='bilinear', align_corners=False)
            # Reduce depth channels về out_channels // 4
            depth_reduced = self.depth_gate[0](depth_feat)   # Conv1x1: depth_ch → C//4
            # RGB reference: trung bình high+low, reduce về C//4 bằng channel grouping
            rgb_avg = (high_feat + low_feat) / 2              # (B, out_channels, H, W)
            B, C, H, W = rgb_avg.shape
            C4 = C // 4
            rgb_ref_fem = rgb_avg.view(B, C4, 4, H, W).mean(dim=2)  # (B, C//4, H, W)
            depth_reduced, _ = self.depth_quality_gate(depth_reduced, rgb_ref_fem)
            # Tạo spatial attention map từ depth
            depth_attn = torch.sigmoid(
                self.depth_gate[1](depth_reduced)   # Conv2d: C//4 → 1 channel
            )
            attn = attn + depth_attn

        high_attn = high_feat * attn
        low_attn  = low_feat  * attn

        concat_feat = torch.cat([high_attn, low_attn], dim=1)
        weights     = self.avg_pool(self.weight_conv(concat_feat))   # (B, 2, 1, 1)

        out = torch.cat([high_attn * weights[:, 0:1],
                         low_attn  * weights[:, 1:2]], dim=1)
        return self.output_conv(out)


# ============================================================================
# Triple Encoder (Res2Net-50 + PVT-v2-b2 RGB + PVT-v2-b1 Depth)
# ============================================================================
class TripleEncoder(nn.Module):
    """
    Triple-stream Encoder với ba backbone song song:
      - RGB-CNN:   Res2Net-50     → channels [256, 512, 1024, 2048]
      - RGB-Trans: PVT-v2-b2     → channels [64,  128,  320,  512]
      - Depth:     PVT-v2-b1     → channels [64,  128,  320,  512]

    PVT-v2-b1 được dùng cho depth stream với trọng số pretrained được
    chuyển đổi từ 3-channel sang 1-channel bằng channel averaging.

    Tại mỗi trong bốn stage, TripleCDFM hợp nhất đặc trưng từ ba stream
    thành một fused feature map duy nhất.
    """
    def __init__(self, pretrained=True):
        super().__init__()

        # RGB-CNN (Res2Net-50)
        self.cnn_backbone  = Res2Net50(pretrained=pretrained)
        self.cnn_channels  = [256, 512, 1024, 2048]

        # RGB-Transformer (PVT-v2-b2)
        self.trans_backbone  = timm.create_model('pvt_v2_b2', pretrained=pretrained, features_only=True)
        self.trans_channels  = [64, 128, 320, 512]

        # Depth stream (PVT-v2-b1, 1-channel input)
        if pretrained:
            print("Loading pretrained PVT-v2-b1 for depth stream...")
            self.depth_backbone = self._load_pretrained_1ch('pvt_v2_b1')
        else:
            self.depth_backbone = timm.create_model(
                'pvt_v2_b1', pretrained=False, features_only=True, in_chans=1)
        self.depth_channels = [64, 128, 320, 512]

        # TripleCDFM tại mỗi stage
        out_ch = [64, 128, 320, 512]
        self.cdfm1 = TripleCDFM(self.cnn_channels[0], self.trans_channels[0], self.depth_channels[0], out_ch[0])
        self.cdfm2 = TripleCDFM(self.cnn_channels[1], self.trans_channels[1], self.depth_channels[1], out_ch[1])
        self.cdfm3 = TripleCDFM(self.cnn_channels[2], self.trans_channels[2], self.depth_channels[2], out_ch[2])
        self.cdfm4 = TripleCDFM(self.cnn_channels[3], self.trans_channels[3], self.depth_channels[3], out_ch[3])

    def _load_pretrained_1ch(self, model_name):
        """
        Chuyển đổi trọng số pretrained 3-channel sang 1-channel
        bằng cách lấy trung bình channel dimension của lớp conv đầu tiên.
        Các lớp còn lại được copy nguyên vẹn nếu shape khớp.
        """
        model_rgb = timm.create_model(model_name, pretrained=True, features_only=True)
        model_1ch = timm.create_model(model_name, pretrained=False, features_only=True, in_chans=1)

        rgb_state = model_rgb.state_dict()
        new_state = model_1ch.state_dict()

        for key in rgb_state.keys():
            if 'weight' in key and len(rgb_state[key].shape) == 4 and rgb_state[key].shape[1] == 3:
                new_state[key] = rgb_state[key].mean(dim=1, keepdim=True)
                print(f"  Adapted {key}: {rgb_state[key].shape} → {new_state[key].shape}")
                break

        for key in rgb_state.keys():
            if key in new_state and rgb_state[key].shape == new_state[key].shape:
                new_state[key] = rgb_state[key]

        model_1ch.load_state_dict(new_state, strict=False)
        print(f"  ✓ Pretrained {model_name} weights adapted for depth stream")
        return model_1ch

    def forward(self, rgb, depth):
        cnn_feat   = self.cnn_backbone(rgb)
        trans_feat = self.trans_backbone(rgb)
        depth_feat = self.depth_backbone(depth)

        c1, w1 = self.cdfm1(cnn_feat[0], trans_feat[0], depth_feat[0])
        c2, w2 = self.cdfm2(cnn_feat[1], trans_feat[1], depth_feat[1])
        c3, w3 = self.cdfm3(cnn_feat[2], trans_feat[2], depth_feat[2])
        c4, w4 = self.cdfm4(cnn_feat[3], trans_feat[3], depth_feat[3])

        fusion_weights = [w1, w2, w3, w4]   # để tính entropy reg loss nếu cần
        return [c1, c2, c3, c4], cnn_feat, trans_feat, depth_feat, fusion_weights


# ============================================================================
# Hàm tiện ích: tính Sobel edge từ depth ground-truth
# ============================================================================
def compute_depth_edge(depth_gt):
    """
    Tính edge map từ depth ground-truth bằng Sobel filter.

    Args:
        depth_gt: (B, 1, H, W), range [0, 1]
    Returns:
        edge_gt:  (B, 1, H, W), normalized về [0, 1]
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=depth_gt.dtype, device=depth_gt.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=depth_gt.dtype, device=depth_gt.device).view(1, 1, 3, 3)

    gx = F.conv2d(depth_gt, sobel_x, padding=1)
    gy = F.conv2d(depth_gt, sobel_y, padding=1)
    magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    # Normalize to [0, 1]
    b = magnitude.shape[0]
    mag_flat = magnitude.view(b, -1)
    max_val  = mag_flat.max(dim=1)[0].view(b, 1, 1, 1).clamp(min=1e-6)
    edge_gt  = magnitude / max_val
    return edge_gt


# ============================================================================
# Complete Model: RDD_COD
# ============================================================================
class RDD_COD(nn.Module):
    """
    RDD-COD: RGB-Depth Dual-stream Camouflaged Object Detection.

    Triple-stream encoder kết hợp Res2Net-50 (CNN), PVT-v2-b2 (RGB Transformer),
    và PVT-v2-b1 (Depth Transformer). Tại mỗi stage, TripleCDFM hợp nhất
    ba stream với depth-guided gating và adaptive weighted fusion.

    BAMTriple phát hiện biên bằng cách kết hợp CNN low-level features,
    Transformer high-level features, và Depth high-level features.

    FEM Decoder tinh chỉnh dự đoán theo hướng từ thô đến chi tiết,
    với edge attention, prediction từ stage trước, và depth features
    được tích hợp trực tiếp vào từng bước giải mã.

    Forward outputs:
        predictions:    tuple(pred_d1, pred_d2, pred_d3, pred_d4) — upsample về input size
        boundary_pred:  edge attention map (B, 1, H, W)
        depth_edge_map: Sobel edge từ depth input (B, 1, H, W) — dùng cho supervision loss
        fusion_weights: list of (B, 3, 1, 1) CDFM weights — dùng cho entropy reg loss
    """
    def __init__(self, n_classes=1, pretrained=True):
        super().__init__()

        self.encoder = TripleEncoder(pretrained=pretrained)
        enc_ch = [64, 128, 320, 512]

        # BAMTriple (cnn=256 từ Res2Net stage1, trans/depth=512 từ PVT stage4)
        self.bam = BAMTriple(cnn_channels=256, trans_channels=512, depth_channels=512)

        # Initial prediction block
        self.initial_block = nn.Sequential(
            ConvBNR(enc_ch[0] + enc_ch[1] + enc_ch[2] + enc_ch[3], 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1)
        )

        # Channel reducers for fused features
        self.reduce1 = Conv1x1(enc_ch[0], 64)
        self.reduce2 = Conv1x1(enc_ch[1], 128)
        self.reduce3 = Conv1x1(enc_ch[2], 256)
        self.reduce4 = Conv1x1(enc_ch[3], 256)

        # FEM Decoder: mỗi stage nhận depth features tương ứng từ PVT-b1
        #   stage4 depth: 512 → fem3
        #   stage3 depth: 320 → fem2
        #   stage2 depth: 128 → fem1
        self.fem3 = FEM(high_channels=256, low_channels=256, out_channels=256, depth_channels=512)
        self.fem2 = FEM(high_channels=256, low_channels=128, out_channels=128, depth_channels=320)
        self.fem1 = FEM(high_channels=128, low_channels=64,  out_channels=64,  depth_channels=128)

        # Output heads
        self.pred1 = nn.Conv2d(64,  n_classes, 1)
        self.pred2 = nn.Conv2d(128, n_classes, 1)
        self.pred3 = nn.Conv2d(256, n_classes, 1)

        for head in [self.pred1, self.pred2, self.pred3]:
            nn.init.constant_(head.bias, 0.01)

    def forward(self, rgb, depth):
        """
        Args:
            rgb:   (B, 3, H, W)
            depth: (B, 1, H, W)

        Returns:
            predictions:    tuple(pred_d1, pred_d2, pred_d3, pred_d4)
            boundary_pred:  (B, 1, H, W)
            depth_edge_map: (B, 1, H, W)
            fusion_weights: list of (B, 3, 1, 1)
        """
        input_size = rgb.shape[2:]

        # ── Encoder ──────────────────────────────────────────────────────────
        fused, cnn_feat, trans_feat, depth_feat, fusion_weights = self.encoder(rgb, depth)
        c1, c2, c3, c4 = fused

        # ── BAM-Triple (Boundary) ─────────────────────────────────────────────
        edge_map  = self.bam(cnn_feat[0], trans_feat[3], depth_feat[3])
        edge_attn = torch.sigmoid(edge_map)

        # ── Initial Prediction ────────────────────────────────────────────────
        size_c4 = c4.shape[2:]
        concat_all = torch.cat([
            c4,
            F.interpolate(c3, size=size_c4, mode='bilinear', align_corners=False),
            F.interpolate(c2, size=size_c4, mode='bilinear', align_corners=False),
            F.interpolate(c1, size=size_c4, mode='bilinear', align_corners=False),
        ], dim=1)
        pred_d4      = self.initial_block(concat_all)
        pred_d4_prob = torch.sigmoid(pred_d4)

        # ── FEM Decoder ───────────────────────────────────────────────────────
        c1f = self.reduce1(c1)
        c2f = self.reduce2(c2)
        c3f = self.reduce3(c3)
        c4f = self.reduce4(c4)

        f3           = self.fem3(c3f, c4f, edge_attn, pred_d4_prob, depth_feat=depth_feat[3])
        pred_d3      = self.pred3(f3)
        pred_d3_prob = torch.sigmoid(pred_d3)

        f2           = self.fem2(c2f, f3, edge_attn, pred_d3_prob, depth_feat=depth_feat[2])
        pred_d2      = self.pred2(f2)
        pred_d2_prob = torch.sigmoid(pred_d2)

        f1           = self.fem1(c1f, f2, edge_attn, pred_d2_prob, depth_feat=depth_feat[1])
        pred_d1      = self.pred1(f1)

        # ── Upsample to input size ────────────────────────────────────────────
        up = lambda t: F.interpolate(t, size=input_size, mode='bilinear', align_corners=False)
        pred_d4_out = up(pred_d4)
        pred_d3_out = up(pred_d3)
        pred_d2_out = up(pred_d2)
        pred_d1_out = up(pred_d1)
        edge_out    = up(edge_attn)

        # Tính depth edge map để dùng làm supervision
        depth_edge_map = compute_depth_edge(depth)

        return (
            (pred_d1_out, pred_d2_out, pred_d3_out, pred_d4_out),
            edge_out,
            depth_edge_map,
            fusion_weights,
        )


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    model = RDD_COD(pretrained=False)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    rgb   = torch.randn(2, 3, 352, 352)
    depth = torch.randn(2, 1, 352, 352)

    predictions, boundary, depth_edge_map, fusion_weights = model(rgb, depth)

    print("\nOutput shapes:")
    for i, pred in enumerate(predictions):
        print(f"  pred_d{i+1}:       {pred.shape}")
    print(f"  boundary:        {boundary.shape}")
    print(f"  depth_edge_map:  {depth_edge_map.shape}")
    print(f"  fusion_weights:  {len(fusion_weights)} stages, each {fusion_weights[0].shape}")