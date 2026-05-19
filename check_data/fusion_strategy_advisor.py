import numpy as np
from pathlib import Path
import json

class FusionStrategyAdvisor:
    """
    Tư vấn chiến lược fusion tối ưu dựa trên correlation analysis
    """
    
    def __init__(self, depth_thermal_corr, rgb_depth_corr, rgb_thermal_corr):
        """
        Args:
            depth_thermal_corr: Pearson correlation giữa Depth và Thermal
            rgb_depth_corr: Pearson correlation giữa RGB và Depth
            rgb_thermal_corr: Pearson correlation giữa RGB và Thermal
        """
        self.d_t_corr = depth_thermal_corr
        self.r_d_corr = rgb_depth_corr
        self.r_t_corr = rgb_thermal_corr
    
    def get_modality_independence(self):
        """
        Tính toán tính độc lập của các modality
        Càng cao = càng bổ sung thông tin
        """
        # Trung bình correlation với các modality khác
        avg_corr = (abs(self.d_t_corr) + 
                   abs(self.r_d_corr) + 
                   abs(self.r_t_corr)) / 3
        
        independence = 1 - avg_corr
        return independence
    
    def get_redundancy_level(self):
        """
        Tính mức độ trùng lặp thông tin
        """
        if self.d_t_corr > 0.7:
            return "HIGH"
        elif self.d_t_corr > 0.4:
            return "MODERATE"
        else:
            return "LOW"
    
    def recommend_modality_combination(self):
        """
        Khuyến nghị cặp modality tốt nhất
        """
        combinations = {
            'RGB+Depth': abs(self.r_d_corr),
            'RGB+Thermal': abs(self.r_t_corr),
            'Depth+Thermal': abs(self.d_t_corr),
            'RGB+Depth+Thermal': np.mean([abs(self.r_d_corr), abs(self.r_t_corr), abs(self.d_t_corr)])
        }
        
        # Sắp xếp theo tính độc lập (correlation thấp nhất = độc lập nhất)
        sorted_combos = sorted(combinations.items(), key=lambda x: x[1])
        
        return sorted_combos
    
    def recommend_fusion_strategy(self):
        """
        Khuyến nghị chiến lược fusion tốt nhất
        """
        strategies = {}
        
        independence = self.get_modality_independence()
        
        # Early Fusion: Ghép tại encoder input
        strategies['Early Fusion'] = {
            'description': 'Concat feature maps at input level',
            'score': independence,  # Cao khi modalities độc lập
            'pros': [
                'Simple to implement',
                'Low computational overhead',
                'Good for early-stage fusion'
            ],
            'cons': [
                'May not learn meaningful cross-modal interactions',
                'Bad with high correlation (redundant)'
            ],
            'recommendation': independence > 0.6
        }
        
        # Mid-level Fusion: Fusion ở giữa mạng
        strategies['Mid-level Fusion'] = {
            'description': 'Fusion at intermediate layers',
            'score': 0.5 * independence + 0.5,  # Balanced
            'pros': [
                'Learns modality-specific and shared features',
                'Better than early fusion',
                'Moderate computational cost'
            ],
            'cons': [
                'More complex to implement',
                'Still may miss some interactions'
            ],
            'recommendation': 0.3 <= independence <= 0.7
        }
        
        # Late Fusion: Fusion ở output
        strategies['Late Fusion'] = {
            'description': 'Process each modality separately, fuse at decision level',
            'score': 1 - independence,  # Cao khi correlation cao
            'pros': [
                'Each modality processed independently',
                'Good when modalities are highly correlated',
                'Easy to add/remove modalities',
                'Robust to modality-specific noise'
            ],
            'cons': [
                'May miss early cross-modal interactions',
                'Need good fusion mechanism at end',
                'Higher computational cost'
            ],
            'recommendation': self.d_t_corr > 0.5
        }
        
        # Cross-modal Attention: Attention giữa các modality
        strategies['Cross-modal Attention'] = {
            'description': 'Each modality attends to others',
            'score': 0.7 if independence > 0.3 else 0.9,
            'pros': [
                'Learns important interactions',
                'Adaptive fusion weights',
                'Works with both similar and diverse modalities',
                'State-of-the-art performance'
            ],
            'cons': [
                'Complex to implement',
                'Higher computational cost',
                'Need more data to train'
            ],
            'recommendation': True  # Always good
        }
        
        # Temporal Fusion (if applicable)
        strategies['Temporal Fusion'] = {
            'description': 'Exploit temporal consistency across modalities',
            'score': 0.6,
            'pros': [
                'Works well with video/sequence data',
                'Exploits temporal coherence'
            ],
            'cons': [
                'Requires temporal data',
                'More complex'
            ],
            'recommendation': False  # Only if temporal data
        }
        
        # Sort by recommendation score
        sorted_strategies = sorted(
            strategies.items(),
            key=lambda x: (x[1]['recommendation'], x[1]['score']),
            reverse=True
        )
        
        return strategies, sorted_strategies
    
    def recommend_architecture(self):
        """
        Khuyến nghị kiến trúc mạng
        """
        independence = self.get_modality_independence()
        
        if self.d_t_corr > 0.7:
            # Depth và Thermal quá tương tự
            return {
                'type': 'Two-stream Network',
                'description': 'Use only two modalities (RGB + Depth or RGB + Thermal)',
                'architecture': '''
                    RGB Stream: ResNet → Feature Extractor
                    Depth/Thermal Stream: Similar architecture
                    Fusion: Late fusion or attention
                    Decoder: Segmentation head
                ''',
                'code': 'architecture_two_stream.py'
            }
        
        elif self.d_t_corr > 0.4:
            # Tương quan vừa phải
            return {
                'type': 'Three-stream Network with Selective Fusion',
                'description': 'Use all three but with careful fusion',
                'architecture': '''
                    RGB Stream: ResNet50 → Conv blocks
                    Depth Stream: Similar architecture
                    Thermal Stream: Similar architecture
                    
                    Feature Extraction:
                    - Level 1: Extract features from each stream
                    - Level 2: Cross-modal attention (optional modality gating)
                    - Level 3: Feature selection (learn which modalities to use)
                    
                    Fusion Strategy:
                    - Mid-level or Late Fusion
                    - Adaptive weights based on feature quality
                    
                    Decoder: Shared or stream-specific decoders
                ''',
                'code': 'architecture_three_stream_selective.py'
            }
        
        else:
            # Các modality rất khác biệt
            return {
                'type': 'Multi-scale Fusion Network',
                'description': 'Full three-stream with early, mid, late fusion',
                'architecture': '''
                    RGB Stream: ResNet50 → Multi-scale features
                    Depth Stream: Multi-scale features
                    Thermal Stream: Multi-scale features
                    
                    Fusion Points:
                    - Early: Input level (optional)
                    - Mid: 2-3 intermediate layers
                    - Late: Before decoder
                    
                    Cross-modal Attention:
                    - Channel attention: What features to focus on
                    - Spatial attention: Where to focus
                    
                    Decoder: Multi-scale decoder with skip connections
                ''',
                'code': 'architecture_three_stream_full.py'
            }
    
    def generate_report(self, output_path=None):
        """
        Tạo báo cáo chi tiết với khuyến nghị
        """
        report = []
        report.append("="*80)
        report.append("🎯 FUSION STRATEGY RECOMMENDATION REPORT")
        report.append("="*80)
        
        # Current Statistics
        report.append("\n📊 CORRELATION ANALYSIS SUMMARY")
        report.append("-"*80)
        report.append(f"Depth ↔ Thermal:  {self.d_t_corr:.4f}")
        report.append(f"RGB ↔ Depth:      {self.r_d_corr:.4f}")
        report.append(f"RGB ↔ Thermal:    {self.r_t_corr:.4f}")
        
        redundancy = self.get_redundancy_level()
        independence = self.get_modality_independence()
        
        report.append(f"\nRedundancy Level:     {redundancy}")
        report.append(f"Modality Independence: {independence:.4f} (0=fully correlated, 1=fully independent)")
        
        # Modality Combination Recommendation
        report.append("\n" + "="*80)
        report.append("💡 MODALITY COMBINATION RECOMMENDATION")
        report.append("="*80)
        
        combos = self.recommend_modality_combination()
        for i, (combo, corr) in enumerate(combos, 1):
            status = "✅ BEST" if i == 1 else "⚠️  MODERATE" if i == 2 else "❌ NOT RECOMMENDED"
            report.append(f"\n{i}. {combo} {status}")
            report.append(f"   Avg Correlation: {corr:.4f}")
            report.append(f"   Independence:    {1-corr:.4f}")
            
            if i == 1:
                if len(combo.split('+')) == 2:
                    report.append(f"   → Recommended: Use only these two modalities")
                else:
                    report.append(f"   → Use if computational resources allow")
        
        # Fusion Strategy Recommendation
        report.append("\n" + "="*80)
        report.append("🔧 FUSION STRATEGY RECOMMENDATION")
        report.append("="*80)
        
        strategies, sorted_strategies = self.recommend_fusion_strategy()
        
        for i, (strategy_name, details) in enumerate(sorted_strategies, 1):
            if details['recommendation']:
                symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                report.append(f"\n{symbol} #{i}. {strategy_name.upper()}")
                report.append(f"    Score: {details['score']:.4f}")
                report.append(f"    Description: {details['description']}")
                report.append(f"\n    PROS:")
                for pro in details['pros']:
                    report.append(f"      ✅ {pro}")
                report.append(f"\n    CONS:")
                for con in details['cons']:
                    report.append(f"      ❌ {con}")
        
        # Architecture Recommendation
        report.append("\n" + "="*80)
        report.append("🏗️  ARCHITECTURE RECOMMENDATION")
        report.append("="*80)
        
        arch = self.recommend_architecture()
        report.append(f"\nType: {arch['type']}")
        report.append(f"Description: {arch['description']}")
        report.append(f"\nArchitecture:")
        report.append(arch['architecture'])
        
        # Implementation Priority
        report.append("\n" + "="*80)
        report.append("📋 IMPLEMENTATION PRIORITY")
        report.append("="*80)
        
        if self.d_t_corr > 0.7:
            report.append("\n1. Start with RGB + Depth baseline")
            report.append("   - Simpler, faster to train")
            report.append("   - Use as reference point")
            report.append("\n2. Optional: Try RGB + Thermal")
            report.append("   - Compare performance to RGB + Depth")
            report.append("   - See if one is significantly better")
            report.append("\n3. Skip: RGB + Depth + Thermal (likely not worth it)")
        
        elif self.d_t_corr > 0.4:
            report.append("\n1. Baseline: RGB + Depth")
            report.append("   - Established modality pair")
            report.append("\n2. Secondary: RGB + Thermal")
            report.append("   - Compare if time allows")
            report.append("\n3. Advanced: RGB + Depth + Thermal with careful fusion")
            report.append("   - Worth trying if baseline is good")
            report.append("   - Use mid-level or late fusion")
            report.append("   - Monitor for overfitting (more parameters)")
        
        else:
            report.append("\n1. Strongly recommended: RGB + Depth + Thermal")
            report.append("   - High diversity in modalities")
            report.append("   - Each modality brings unique information")
            report.append("\n2. Fusion strategy:")
            report.append("   - Use Late Fusion or Cross-modal Attention")
            report.append("   - Early Fusion less recommended")
            report.append("\n3. Consider:")
            report.append("   - Multi-task learning (separate decoders per modality)")
            report.append("   - Feature alignment before fusion")
        
        # Key Metrics to Monitor
        report.append("\n" + "="*80)
        report.append("📈 KEY METRICS TO MONITOR")
        report.append("="*80)
        report.append("\nWhen training:")
        report.append("  1. Validation mAP/IoU (final performance)")
        report.append("  2. Per-modality contribution (ablation study)")
        report.append("  3. Inference time (speed vs accuracy trade-off)")
        report.append("  4. GPU memory (computational efficiency)")
        report.append("\nAblation Study Structure:")
        report.append("  • Baseline 1: RGB only")
        report.append("  • Baseline 2: RGB + Depth")
        report.append("  • Baseline 3: RGB + Thermal")
        report.append("  • Experiment: RGB + Depth + Thermal")
        report.append("\nCompute improvement as:")
        report.append("  mAP gain = (RGB+D+T performance) - max(RGB+D, RGB+T)")
        
        # Final Verdict
        report.append("\n" + "="*80)
        report.append("🎯 FINAL VERDICT")
        report.append("="*80)
        
        if self.d_t_corr > 0.7:
            verdict = "❌ DO NOT use all three modalities"
            reason = "Depth and Thermal are too similar (high redundancy)"
        elif self.d_t_corr > 0.4:
            verdict = "⚠️  CONDITIONALLY use all three"
            reason = "Moderate complementarity; test carefully"
        else:
            verdict = "✅ STRONGLY RECOMMENDED to use all three"
            reason = "High complementarity; good information diversity"
        
        report.append(f"\n{verdict}")
        report.append(f"Reason: {reason}")
        report.append(f"\nDepth-Thermal Correlation: {self.d_t_corr:.4f}")
        report.append(f"Interpretation: {'High redundancy (>0.7)' if self.d_t_corr > 0.7 else 'Moderate redundancy (0.4-0.7)' if self.d_t_corr > 0.4 else 'Low redundancy (<0.4)'}")
        
        report.append("\n" + "="*80)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"✅ Report saved to {output_path}")
        
        return report_text


# Example usage
if __name__ == "__main__":
    # Ví dụ 1: High correlation
    print("\n" + "="*80)
    print("EXAMPLE 1: High Correlation (Depth-Thermal = 0.85)")
    print("="*80)
    
    advisor = FusionStrategyAdvisor(
        depth_thermal_corr=0.85,
        rgb_depth_corr=0.45,
        rgb_thermal_corr=0.48
    )
    
    report = advisor.generate_report()
    print(report)
    
    # Ví dụ 2: Moderate correlation
    print("\n" + "="*80)
    print("EXAMPLE 2: Moderate Correlation (Depth-Thermal = 0.55)")
    print("="*80)
    
    advisor = FusionStrategyAdvisor(
        depth_thermal_corr=0.55,
        rgb_depth_corr=0.35,
        rgb_thermal_corr=0.38
    )
    
    report = advisor.generate_report()
    print(report)
    
    # Ví dụ 3: Low correlation
    print("\n" + "="*80)
    print("EXAMPLE 3: Low Correlation (Depth-Thermal = 0.25)")
    print("="*80)
    
    advisor = FusionStrategyAdvisor(
        depth_thermal_corr=0.25,
        rgb_depth_corr=0.32,
        rgb_thermal_corr=0.28
    )
    
    report = advisor.generate_report()
    print(report)