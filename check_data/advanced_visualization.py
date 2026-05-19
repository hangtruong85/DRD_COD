import os
import numpy as np
import cv2
from pathlib import Path
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class AdvancedVisualization:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.train_path = self.root_path / 'train'
    
    def load_image(self, path, as_gray=False):
        try:
            if as_gray:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            return img
        except:
            return None
    
    def normalize_image(self, img):
        if img is None:
            return None
        img = img.astype(np.float32)
        img_min = img.min()
        img_max = img.max()
        if img_max == img_min:
            return np.zeros_like(img)
        return (img - img_min) / (img_max - img_min)
    
    def resize_to_common(self, img1, img2):
        """Resize hai ảnh về cùng kích thước"""
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        return img1[:h, :w], img2[:h, :w]
    
    def visualize_sample_pairs(self, num_samples=5):
        """Visualize một số cặp ảnh từ các modality khác nhau"""
        depth_path = self.train_path / 'depth'
        thermal_path = self.train_path / 'thermal'
        rgb_path = self.train_path / 'images'
        
        depth_files = sorted(list(depth_path.glob('*.png')))[:num_samples]
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        
        for idx, depth_file in enumerate(depth_files):
            # Get corresponding files
            name = depth_file.stem
            thermal_file = thermal_path / f"{name}.png"
            rgb_file = rgb_path / f"{name}.jpg"
            
            # Load images
            depth = self.load_image(depth_file, as_gray=True)
            thermal = self.load_image(thermal_file, as_gray=True)
            rgb = self.load_image(rgb_file, as_gray=False)
            
            if rgb is not None and len(rgb.shape) == 3:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            
            # Calculate correlations
            depth_norm = self.normalize_image(depth)
            thermal_norm = self.normalize_image(thermal)
            rgb_norm = self.normalize_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)) if len(rgb.shape) == 3 else self.normalize_image(rgb)
            
            d_t_corr = np.corrcoef(depth_norm.flatten(), thermal_norm.flatten())[0, 1]
            r_d_corr = np.corrcoef(rgb_norm.flatten(), depth_norm.flatten())[0, 1]
            r_t_corr = np.corrcoef(rgb_norm.flatten(), thermal_norm.flatten())[0, 1]
            
            # Plot
            ax = axes[idx, 0]
            ax.imshow(rgb, cmap='gray' if len(rgb.shape) == 2 else None)
            ax.set_title(f'Sample {idx+1}: RGB')
            ax.axis('off')
            
            ax = axes[idx, 1]
            ax.imshow(depth_norm, cmap='viridis')
            ax.set_title(f'Depth\n(vs RGB: {r_d_corr:.3f})')
            ax.axis('off')
            
            ax = axes[idx, 2]
            ax.imshow(thermal_norm, cmap='hot')
            ax.set_title(f'Thermal\n(vs RGB: {r_t_corr:.3f})')
            ax.axis('off')
            
            ax = axes[idx, 3]
            # Heatmap showing difference
            diff = np.abs(depth_norm - thermal_norm)
            ax.imshow(diff, cmap='RdYlBu_r')
            ax.set_title(f'Depth-Thermal Diff\n(Corr: {d_t_corr:.3f})')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('./sample_pairs_visualization.png', dpi=300, bbox_inches='tight')
        print("✅ Sample pairs visualization saved")
        plt.show()
    
    def create_correlation_heatmap(self):
        """Tạo heatmap correlation"""
        depth_path = self.train_path / 'depth'
        thermal_path = self.train_path / 'thermal'
        rgb_path = self.train_path / 'images'
        
        depth_files = sorted(list(depth_path.glob('*.png')))[:100]  # Sample 100
        
        correlations_matrix = []
        
        for depth_file in tqdm(depth_files, desc="Computing correlations"):
            name = depth_file.stem
            thermal_file = thermal_path / f"{name}.png"
            rgb_file = rgb_path / f"{name}.jpg"
            
            depth = self.normalize_image(self.load_image(depth_file, as_gray=True))
            thermal = self.normalize_image(self.load_image(thermal_file, as_gray=True))
            rgb = self.load_image(rgb_file, as_gray=False)
            
            if rgb is not None and len(rgb.shape) == 3:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            rgb = self.normalize_image(rgb)
            
            # Resize
            depth, thermal = self.resize_to_common(depth, thermal)
            depth, rgb = self.resize_to_common(depth, rgb)
            thermal, rgb = self.resize_to_common(thermal, rgb)
            
            # Calculate correlations
            depth_flat = depth.flatten()
            thermal_flat = thermal.flatten()
            rgb_flat = rgb.flatten()
            
            d_t_corr = np.corrcoef(depth_flat, thermal_flat)[0, 1]
            d_r_corr = np.corrcoef(depth_flat, rgb_flat)[0, 1]
            t_r_corr = np.corrcoef(thermal_flat, rgb_flat)[0, 1]
            
            correlations_matrix.append([d_t_corr, d_r_corr, t_r_corr])
        
        corr_array = np.array(correlations_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        labels = ['Depth-Thermal', 'Depth-RGB', 'Thermal-RGB']
        mean_corr = np.mean(corr_array, axis=0)
        std_corr = np.std(corr_array, axis=0)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax.bar(labels, mean_corr, yerr=std_corr, capsize=10, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Pearson Correlation', fontweight='bold', fontsize=12)
        ax.set_title('Average Correlation between Modalities\n(with std error bars)', fontweight='bold', fontsize=14)
        ax.set_ylim([-0.2, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, mean_corr, std_corr)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('./correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✅ Correlation heatmap saved")
        plt.show()
        
        return corr_array
    
    def analyze_complementarity(self):
        """Phân tích tính bổ sung của các modality"""
        depth_path = self.train_path / 'depth'
        thermal_path = self.train_path / 'thermal'
        rgb_path = self.train_path / 'images'
        
        depth_files = sorted(list(depth_path.glob('*.png')))[:50]
        
        metrics = {
            'depth_entropy': [],
            'thermal_entropy': [],
            'rgb_entropy': [],
            'depth_contrast': [],
            'thermal_contrast': [],
            'rgb_contrast': [],
        }
        
        for depth_file in tqdm(depth_files, desc="Analyzing complementarity"):
            name = depth_file.stem
            thermal_file = thermal_path / f"{name}.png"
            rgb_file = rgb_path / f"{name}.jpg"
            
            depth = self.normalize_image(self.load_image(depth_file, as_gray=True))
            thermal = self.normalize_image(self.load_image(thermal_file, as_gray=True))
            rgb = self.load_image(rgb_file, as_gray=False)
            
            if rgb is not None and len(rgb.shape) == 3:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            rgb = self.normalize_image(rgb)
            
            # Calculate entropy
            def entropy(img):
                hist, _ = np.histogram(img.flatten() * 255, bins=256, range=[0, 256])
                hist = hist / hist.sum()
                return -np.sum(hist * np.log2(hist + 1e-10))
            
            metrics['depth_entropy'].append(entropy(depth))
            metrics['thermal_entropy'].append(entropy(thermal))
            metrics['rgb_entropy'].append(entropy(rgb))
            
            # Calculate contrast (standard deviation)
            metrics['depth_contrast'].append(np.std(depth))
            metrics['thermal_contrast'].append(np.std(thermal))
            metrics['rgb_contrast'].append(np.std(rgb))
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Entropy
        ax = axes[0]
        entropy_data = [metrics['depth_entropy'], metrics['thermal_entropy'], metrics['rgb_entropy']]
        bp = ax.boxplot(entropy_data, labels=['Depth', 'Thermal', 'RGB'], patch_artist=True)
        colors = ['#FF6B6B', '#FFB84D', '#4ECDC4']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Entropy', fontweight='bold')
        ax.set_title('Information Entropy by Modality', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Contrast
        ax = axes[1]
        contrast_data = [metrics['depth_contrast'], metrics['thermal_contrast'], metrics['rgb_contrast']]
        bp = ax.boxplot(contrast_data, labels=['Depth', 'Thermal', 'RGB'], patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Contrast (Std Dev)', fontweight='bold')
        ax.set_title('Contrast by Modality', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./complementarity_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Complementarity analysis saved")
        plt.show()
    
    def run(self, root_path):
        print("\n🚀 Starting Advanced Visualization...")
        
        print("\n📊 Creating sample pairs visualization...")
        self.visualize_sample_pairs(num_samples=5)
        
        print("\n📊 Creating correlation heatmap...")
        corr_data = self.create_correlation_heatmap()
        
        print("\n📊 Analyzing complementarity...")
        self.analyze_complementarity()
        
        print("\n✅ All visualizations completed!")


if __name__ == "__main__":
    import sys
    root_path = sys.argv[1] if len(sys.argv) > 1 else "./data"
    
    viz = AdvancedVisualization(root_path)
    viz.run(root_path)