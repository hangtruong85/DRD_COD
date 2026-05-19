import os
import numpy as np
import cv2
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

class CorrelationAnalyzer:
    def __init__(self, root_path):
        self.train_path = Path(root_path)
        
    def load_image(self, path, as_gray=False):
        """Load image từ path"""
        try:
            if as_gray:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def normalize_image(self, img):
        """Normalize image về range [0, 1]"""
        if img is None:
            return None
        img = img.astype(np.float32)
        img_min = img.min()
        img_max = img.max()
        if img_max == img_min:
            return np.zeros_like(img)
        return (img - img_min) / (img_max - img_min)
    
    def calculate_ssim(self, img1, img2):
        """Tính SSIM giữa hai ảnh"""
        if img1 is None or img2 is None:
            return None
        
        # Resize về cùng kích thước
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
        
        # Normalize
        img1 = self.normalize_image(img1)
        img2 = self.normalize_image(img2)
        
        # Calculate SSIM
        try:
            ssim_value = ssim(img1, img2, data_range=1.0)
            return ssim_value
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return None
    
    def calculate_pearson_correlation(self, img1, img2):
        """Tính tương quan Pearson"""
        if img1 is None or img2 is None:
            return None
        
        # Resize về cùng kích thước
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
        
        # Flatten và normalize
        img1_flat = self.normalize_image(img1).flatten()
        img2_flat = self.normalize_image(img2).flatten()
        
        try:
            corr, _ = pearsonr(img1_flat, img2_flat)
            return corr
        except Exception as e:
            return None
    
    def calculate_spearman_correlation(self, img1, img2):
        """Tính tương quan Spearman"""
        if img1 is None or img2 is None:
            return None
        
        # Resize về cùng kích thước
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
        
        # Flatten và normalize
        img1_flat = self.normalize_image(img1).flatten()
        img2_flat = self.normalize_image(img2).flatten()
        
        try:
            corr, _ = spearmanr(img1_flat, img2_flat)
            return corr
        except Exception as e:
            return None
    
    def calculate_mse(self, img1, img2):
        """Tính Mean Squared Error"""
        if img1 is None or img2 is None:
            return None
        
        # Resize về cùng kích thước
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
        
        # Normalize
        img1 = self.normalize_image(img1)
        img2 = self.normalize_image(img2)
        
        mse = np.mean((img1 - img2) ** 2)
        return mse
    
    def analyze_split(self, split_path, split_name='train'):
        """Phân tích một split (train/val)"""
        depth_path = split_path / 'depth'
        thermal_path = split_path / 'thermal'
        rgb_path = split_path / 'images'
        
        if not all([depth_path.exists(), thermal_path.exists(), rgb_path.exists()]):
            print(f"❌ Missing directories in {split_name}")
            return None
        
        results = {
            'ssim_depth_thermal': [],
            'pearson_depth_thermal': [],
            'spearman_depth_thermal': [],
            'mse_depth_thermal': [],
            'ssim_rgb_depth': [],
            'pearson_rgb_depth': [],
            'mse_rgb_depth': [],
            'ssim_rgb_thermal': [],
            'pearson_rgb_thermal': [],
            'mse_rgb_thermal': [],
        }
        
        depth_files = sorted(list(depth_path.glob('*.png')))
        thermal_files = sorted(list(thermal_path.glob('*.png')))
        rgb_files = sorted(list(rgb_path.glob('*.jpg')))
        
        print(f"\n📊 Analyzing {split_name} split ({len(depth_files)} samples)...")
        
        for i, (depth_file, thermal_file, rgb_file) in enumerate(tqdm(
            zip(depth_files, thermal_files, rgb_files), 
            total=len(depth_files),
            desc=f"{split_name}"
        )):
            # Load ảnh
            depth = self.load_image(depth_file, as_gray=True)
            thermal = self.load_image(thermal_file, as_gray=True)
            rgb = self.load_image(rgb_file, as_gray=False)
            
            # Convert RGB to grayscale
            if rgb is not None and len(rgb.shape) == 3:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            
            # Depth vs Thermal
            results['ssim_depth_thermal'].append(self.calculate_ssim(depth, thermal))
            results['pearson_depth_thermal'].append(self.calculate_pearson_correlation(depth, thermal))
            results['spearman_depth_thermal'].append(self.calculate_spearman_correlation(depth, thermal))
            results['mse_depth_thermal'].append(self.calculate_mse(depth, thermal))
            
            # RGB vs Depth
            results['ssim_rgb_depth'].append(self.calculate_ssim(rgb, depth))
            results['pearson_rgb_depth'].append(self.calculate_pearson_correlation(rgb, depth))
            results['mse_rgb_depth'].append(self.calculate_mse(rgb, depth))
            
            # RGB vs Thermal
            results['ssim_rgb_thermal'].append(self.calculate_ssim(rgb, thermal))
            results['pearson_rgb_thermal'].append(self.calculate_pearson_correlation(rgb, thermal))
            results['mse_rgb_thermal'].append(self.calculate_mse(rgb, thermal))
        
        return results
    
    def print_statistics(self, results, split_name):
        """In ra thống kê"""
        if results is None:
            return
        
        print(f"\n{'='*70}")
        print(f"📈 Statistics for {split_name.upper()}")
        print(f"{'='*70}")
        
        metrics = {
            'Depth vs Thermal (SSIM)': 'ssim_depth_thermal',
            'Depth vs Thermal (Pearson)': 'pearson_depth_thermal',
            'Depth vs Thermal (Spearman)': 'spearman_depth_thermal',
            'Depth vs Thermal (MSE)': 'mse_depth_thermal',
            'RGB vs Depth (SSIM)': 'ssim_rgb_depth',
            'RGB vs Depth (Pearson)': 'pearson_rgb_depth',
            'RGB vs Depth (MSE)': 'mse_rgb_depth',
            'RGB vs Thermal (SSIM)': 'ssim_rgb_thermal',
            'RGB vs Thermal (Pearson)': 'pearson_rgb_thermal',
            'RGB vs Thermal (MSE)': 'mse_rgb_thermal',
        }
        
        for name, key in metrics.items():
            values = [v for v in results[key] if v is not None]
            if values:
                print(f"\n{name}:")
                print(f"  Mean:     {np.mean(values):>8.4f}")
                print(f"  Median:   {np.median(values):>8.4f}")
                print(f"  Std:      {np.std(values):>8.4f}")
                print(f"  Min:      {np.min(values):>8.4f}")
                print(f"  Max:      {np.max(values):>8.4f}")
    
    def visualize_results(self, train_results):
        """Visualize kết quả"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Correlation Analysis between Modalities', fontsize=16, fontweight='bold')
        
        # SSIM Comparison
        ax = axes[0, 0]
        ssim_data = {
            'Depth vs\nThermal': [v for v in train_results['ssim_depth_thermal'] if v is not None],
            'RGB vs\nDepth': [v for v in train_results['ssim_rgb_depth'] if v is not None],
            'RGB vs\nThermal': [v for v in train_results['ssim_rgb_thermal'] if v is not None],
        }
        bp = ax.boxplot([v for v in ssim_data.values()], labels=ssim_data.keys(), patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_ylabel('SSIM Score', fontweight='bold')
        ax.set_title('SSIM Comparison (Train Set)')
        ax.grid(axis='y', alpha=0.3)
        
        # Pearson Correlation
        ax = axes[0, 1]
        pearson_data = {
            'Depth vs\nThermal': [v for v in train_results['pearson_depth_thermal'] if v is not None],
            'RGB vs\nDepth': [v for v in train_results['pearson_rgb_depth'] if v is not None],
            'RGB vs\nThermal': [v for v in train_results['pearson_rgb_thermal'] if v is not None],
        }
        bp = ax.boxplot([v for v in pearson_data.values()], labels=pearson_data.keys(), patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
        ax.set_ylabel('Pearson Correlation', fontweight='bold')
        ax.set_title('Pearson Correlation Comparison (Train Set)')
        ax.grid(axis='y', alpha=0.3)
        
        # MSE Comparison
        ax = axes[1, 0]
        mse_data = {
            'Depth vs\nThermal': [v for v in train_results['mse_depth_thermal'] if v is not None],
            'RGB vs\nDepth': [v for v in train_results['mse_rgb_depth'] if v is not None],
            'RGB vs\nThermal': [v for v in train_results['mse_rgb_thermal'] if v is not None],
        }
        bp = ax.boxplot([v for v in mse_data.values()], labels=mse_data.keys(), patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightyellow')
        ax.set_ylabel('MSE Value', fontweight='bold')
        ax.set_title('MSE Comparison (Train Set)')
        ax.grid(axis='y', alpha=0.3)
        
        # Distribution of Depth vs Thermal Correlation
        ax = axes[1, 1]
        depth_thermal_corr = [v for v in train_results['pearson_depth_thermal'] if v is not None]
        ax.hist(depth_thermal_corr, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(depth_thermal_corr), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(depth_thermal_corr):.3f}')
        ax.set_xlabel('Pearson Correlation', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Distribution of Depth-Thermal Correlation')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./correlation_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✅ Visualization saved to correlation_analysis.png")
        plt.show()
    
    def generate_report(self, train_results):
        """Tạo báo cáo chi tiết"""
        report = []
        report.append("="*80)
        report.append("MULTIMODAL CORRELATION ANALYSIS REPORT")
        report.append("="*80)
        
        # Analysis for Train
        if train_results:
            report.append("\n📊 TRAIN SET ANALYSIS")
            report.append("-"*80)
            
            depth_thermal_pearson = [v for v in train_results['pearson_depth_thermal'] if v is not None]
            report.append(f"\n🔍 Depth-Thermal Correlation:")
            report.append(f"   Mean Pearson: {np.mean(depth_thermal_pearson):.4f}")
            report.append(f"   Mean SSIM:    {np.mean([v for v in train_results['ssim_depth_thermal'] if v is not None]):.4f}")
            
            if np.mean(depth_thermal_pearson) > 0.7:
                report.append(f"   ⚠️  HIGH correlation! Depth and Thermal are very similar.")
                report.append(f"      Using both may not add much complementary information.")
            elif np.mean(depth_thermal_pearson) > 0.4:
                report.append(f"   ✓ MODERATE correlation. They provide some complementary information.")
            else:
                report.append(f"   ✓ LOW correlation. They provide diverse information.")
            
            rgb_depth_pearson = [v for v in train_results['pearson_rgb_depth'] if v is not None]
            rgb_thermal_pearson = [v for v in train_results['pearson_rgb_thermal'] if v is not None]
            
            report.append(f"\n🔍 RGB Correlation with other modalities:")
            report.append(f"   RGB-Depth:  {np.mean(rgb_depth_pearson):.4f}")
            report.append(f"   RGB-Thermal: {np.mean(rgb_thermal_pearson):.4f}")
     
        # Recommendations
        report.append("\n" + "="*80)
        report.append("💡 RECOMMENDATIONS")
        report.append("="*80)
        
        depth_thermal_corr = np.mean([v for v in train_results['pearson_depth_thermal'] if v is not None])
        
        if depth_thermal_corr > 0.7:
            report.append("\n❌ RGB + Depth + Thermal: NOT RECOMMENDED")
            report.append("   Reason: Depth and Thermal have high correlation (redundant information)")
            report.append("\n✅ BETTER OPTIONS:")
            report.append("   • RGB + Depth (simpler, faster, less redundancy)")
            report.append("   • RGB + Thermal (if Thermal is more important for your task)")
            
        elif depth_thermal_corr > 0.4:
            report.append("\n⚠️  RGB + Depth + Thermal: MIGHT WORK")
            report.append("   Moderate correlation suggests some complementary information")
            report.append("   But consider computational cost vs performance gain")
            report.append("\n✅ BETTER OPTIONS:")
            report.append("   • RGB + Depth (good baseline)")
            report.append("   • Experiment with Feature Fusion strategies (early, mid, late fusion)")
        else:
            report.append("\n✅ RGB + Depth + Thermal: RECOMMENDED")
            report.append("   Low correlation indicates good complementary information")
            report.append("   All three modalities should contribute diverse features")
            report.append("\n💡 IMPLEMENTATION TIPS:")
            report.append("   • Use cross-modal attention mechanisms")
            report.append("   • Consider late fusion for maximum information retention")
            report.append("   • Monitor computational cost vs accuracy improvement")
        
        report.append("\n" + "="*80)
        
        report_text = "\n".join(report)
        return report_text
    
    def run(self, root_path):
        """Chạy phân tích hoàn toàn"""
        print("\n🚀 Starting Correlation Analysis...")
        
        train_results = self.analyze_split(self.train_path, 'train')
        
        # Print statistics
        if train_results:
            self.print_statistics(train_results, 'train')
        
        # Visualize
        if train_results:
            self.visualize_results(train_results)
        
        # Generate report
        if train_results:
            report = self.generate_report(train_results)
            print("\n" + report)
            
            # Save report
            with open('./correlation_report.txt', 'w') as f:
                f.write(report)
            print("\n✅ Report saved to correlation_report.txt")
        
        return train_results 


if __name__ == "__main__":
    import sys
    
    # Default path
    root_path = sys.argv[1] if len(sys.argv) > 1 else "./data"
    
    analyzer = CorrelationAnalyzer(root_path)
    analyzer.run(root_path)