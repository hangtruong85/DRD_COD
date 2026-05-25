# DRD-COD: A Depth-Guided RGB Dual-Stream Network for Camouflaged Object Detection

## Abstract

Camouflaged object detection (COD) is a challenging visual recognition task in which objects blend into their surroundings through highly similar textures, colors, and patterns, making them difficult to detect using appearance cues alone. Existing COD methods primarily rely on RGB images, while geometric structural information remains largely underexplored. Although monocular depth maps provide useful geometric discontinuities that may reveal camouflage boundaries, directly integrating monocular depth into COD is challenging due to the noise and artifacts introduced by depth estimation, especially around low-contrast object boundaries.

To address this issue, we propose **DRD-COD**, a triple-stream RGB-Depth network for camouflaged object detection that effectively exploits monocular depth without requiring depth sensors or depth-annotated training data. DRD-COD employs three parallel encoders to capture complementary representations: a CNN encoder for local texture details, a Transformer encoder for global semantic context, and a depth encoder for geometric structural cues. To effectively integrate these heterogeneous features, we introduce a **Depth-guided Cross-stream Fusion Module (DCFM)**, which performs depth-guided asymmetric fusion between CNN, Transformer, and depth features through adaptive multi-branch aggregation. In addition, a lightweight **Depth Quality Gate (DQG)** is embedded throughout the network to dynamically evaluate depth reliability and suppress noisy depth responses when geometric cues are inconsistent with RGB representations. To further enhance camouflage boundary localization, we propose a **Triple-stream Boundary Attention Module (TBAM)** that jointly exploits low-level texture details, high-level semantic information, and depth geometry to generate explicit boundary guidance for the decoder.

Extensive experiments on four COD benchmarks, including CAMO, COD10K, NC4K, and MHCD-Seg, demonstrate that DRD-COD achieves state-of-the-art performance, outperforming existing RGB-Depth methods and remaining competitive with RGB-only approaches using substantially larger backbones or foundation-model pretraining.

## Citation

If you find this work useful, please cite:

```bibtex
@article{truong2026drdcod,
  title   = {DRD-COD: A Depth-Guided RGB Dual-Stream Network for
             Camouflaged Object Detection},
  author  = {Truong, Thi-Thu-Hang and others},
  journal = {ResearchGate preprint},
  year    = {2026}
}
```
