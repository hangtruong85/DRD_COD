import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DatasetWithDepth(Dataset):
    """
    Dataset for merged CAMO + COD10K-v3 with RGB + Depth + Mask support
    Uses ReplayCompose to ensure synchronized augmentation
    
    Directory structure:
        camo_cod10k/
        ├── Train/
        │   ├── images/          # RGB images (.jpg, .png)
        │   ├── GT/              # Object segmentation masks (.png)
        │   └── depth/           # Depth maps (.png)
        ├── Val/
        │   ├── images/
        │   ├── GT/
        │   └── depth/
        └── Test/
            ├── images/
            ├── GT/
            └── depth/
    
    Returns:
        rgb: (3, H, W) - Normalized with ImageNet mean/std
        depth: (1, H, W) - Normalized to [0, 1]
        mask: (1, H, W) - Binary {0, 1}
    
    Combines:
    - CAMO: 1000 train + 200 val + 250 test = 1450 images
    - COD10K-v3: 6000 train + 1200 val + 250 test = 7450 images
    - Total: 7000 train + 1400 val + 500 test = 8900 images
    """
    
    def __init__(self, root, split="train", img_size=256, augment=True, 
                 use_depth=True, logger=None):
        """
        Args:
            root: Path to camo_cod10k root directory
            split: "train", "val", or "test"
            img_size: Target image size (default 256)
            augment: Whether to apply augmentation (only for train split)
            use_depth: Whether to load depth maps
            logger: Logger instance (optional)
        """
        
        # Convert split name
        if split == "val":
            split_dir = "Val"
        elif split == "test":
            split_dir = "Test"
        else:  # "train"
            split_dir = "Train"
        
        self.img_dir = os.path.join(root, split_dir, "images")
        self.mask_dir = os.path.join(root, split_dir, "GT")
        self.depth_dir = os.path.join(root, split_dir, "depth")
        
        self.use_depth = use_depth
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.split = split
        
        # Get image paths (support both .jpg and .png)
        self.img_paths = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg")) +
            glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        assert len(self.img_paths) > 0, f"No images found in {self.img_dir}"
        
        # Build mask paths
        self.mask_paths = []
        for ip in self.img_paths:
            basename = os.path.splitext(os.path.basename(ip))[0]
            mask_path = os.path.join(self.mask_dir, basename + ".png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            self.mask_paths.append(mask_path)
        
        # Build depth paths
        if self.use_depth:
            if not os.path.exists(self.depth_dir):
                if logger:
                    logger.warning(f"Depth directory not found: {self.depth_dir}")
                    logger.warning("Depth loading disabled!")
                else:
                    print(f"WARNING: Depth directory not found: {self.depth_dir}")
                    print("Depth loading disabled!")
                self.use_depth = False
            else:
                self.depth_paths = []
                for ip in self.img_paths:
                    basename = os.path.splitext(os.path.basename(ip))[0]
                    depth_path = os.path.join(self.depth_dir, basename + ".png")
                    if not os.path.exists(depth_path):
                        raise FileNotFoundError(f"Depth not found: {depth_path}")
                    self.depth_paths.append(depth_path)
        
        # Setup transforms using ReplayCompose for synchronization
        if self.augment:
            self.geometric_aug = A.ReplayCompose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.4
                ),
                A.Rotate(limit=10, p=0.3),
            ])
            self.color_aug = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(p=0.1),
            ])
        else:
            self.geometric_aug = A.ReplayCompose([
                A.Resize(img_size, img_size),
            ])
            self.color_aug = None
        
        self.rgb_normalize = A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        )
        
        # Logging
        log_msg = f"[CAMO+COD10K Dataset] {split_dir}: images={len(self.img_paths)}, masks={len(self.mask_paths)}"
        if self.use_depth:
            log_msg += f", depth={len(self.depth_paths)}"
        else:
            log_msg += ", depth=DISABLED"
        
        print(log_msg)
        if logger:
            logger.info(log_msg)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load RGB image
        img = cv2.imread(self.img_paths[idx])
        if img is None:
            raise RuntimeError(f"Failed to load image: {self.img_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask (GT) - keep original 0-255 range for augmentation
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {self.mask_paths[idx]}")
        
        # Load depth
        if self.use_depth:
            depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_GRAYSCALE)
            if depth is None:
                raise RuntimeError(f"Failed to load depth: {self.depth_paths[idx]}")
        else:
            # Fallback: create depth from grayscale image
            depth = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply geometric augmentation to RGB and record transforms
        geometric_result = self.geometric_aug(image=img)
        img = geometric_result['image']
        
        # Apply SAME geometric transforms to depth and mask using replay
        depth = A.ReplayCompose.replay(geometric_result['replay'], image=depth)['image']
        mask = A.ReplayCompose.replay(geometric_result['replay'], image=mask)['image']
        
        # Apply color augmentation to RGB only
        if self.color_aug is not None:
            img = self.color_aug(image=img)['image']
        
        # Normalize RGB
        img = self.rgb_normalize(image=img)['image']
        
        # Normalize depth to [0, 1]
        depth = depth.astype("float32") / 255.0
        
        # Binarize mask (mask is in 0-255 range after resize)
        # Both CAMO and COD10K use 255 for object, 0 for background
        mask = (mask > 127).astype("float32")
        
        # Convert to tensors
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)
        depth = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
        return img, depth, mask


class DatasetCAMOCOD10K(Dataset):
    """
    Simplified Dataset for merged CAMO + COD10K-v3 with RGB + Mask support (no depth)
    
    Directory structure:
        camo_cod10k/
        ├── Train/
        │   ├── images/          # RGB images (.jpg, .png)
        │   ├── GT/              # Object segmentation masks (.png)
        │   └── depth/           # Depth maps (optional)
        ├── Val/
        │   ├── images/
        │   ├── GT/
        │   └── depth/
        └── Test/
            ├── images/
            ├── GT/
            └── depth/
    
    Returns:
        rgb: (3, H, W) - Normalized with ImageNet mean/std
        mask: (1, H, W) - Binary {0, 1}
    
    Combines CAMO (1450 images) and COD10K-v3 (7450 images) = 8900 total
    """
    
    def __init__(self, root, split="train", img_size=256, augment=True, logger=None):
        """
        Args:
            root: Path to camo_cod10k root directory
            split: "train", "val", or "test"
            img_size: Target image size (default 256)
            augment: Whether to apply augmentation (only for train split)
            logger: Logger instance (optional)
        """
        
        # Convert split name
        if split == "val":
            split_dir = "Val"
        elif split == "test":
            split_dir = "Test"
        else:  # "train"
            split_dir = "Train"
        
        self.img_dir = os.path.join(root, split_dir, "images")
        self.mask_dir = os.path.join(root, split_dir, "GT")
        self.img_size = img_size
        self.split = split
        
        # Get image paths (support both .jpg and .png)
        self.img_paths = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg")) +
            glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        assert len(self.img_paths) > 0, f"No images found in {self.img_dir}"
        
        # Build mask paths
        self.mask_paths = []
        for ip in self.img_paths:
            basename = os.path.splitext(os.path.basename(ip))[0]
            mask_path = os.path.join(self.mask_dir, basename + ".png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            self.mask_paths.append(mask_path)
        
        # Setup transforms
        if augment and split == "train":
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.4
                ),
                A.Rotate(limit=10, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(p=0.1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        # Logging
        log_msg = f"[CAMO+COD10K Dataset] {split_dir}: images={len(self.img_paths)}, masks={len(self.mask_paths)}"
        print(log_msg)
        if logger:
            logger.info(log_msg)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load RGB image
        img = cv2.imread(self.img_paths[idx])
        if img is None:
            raise RuntimeError(f"Failed to load image: {self.img_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask (GT)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {self.mask_paths[idx]}")
        
        # Binarize mask (both CAMO and COD10K use 255 for object, 0 for background)
        mask = (mask > 127).astype("float32")
        
        # Apply transforms
        sample = self.transform(image=img, mask=mask)
        img = sample["image"]
        mask = sample["mask"].unsqueeze(0)  # Add channel dimension: (1, H, W)
        
        return img, mask