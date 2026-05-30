import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from data_utils import (load_and_stack_sar, load_and_stack_optical, load_cloud_mask, load_mask,
    apply_speckle_filter, normalize_sar, normalize_optical, get_cloud_coverage 
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class FloodDataset(Dataset):
    def __init__(self, manifest_path, split='train', augment=True):
        """
        Custom dataset for flood mapping using Sentinel-2 optical and Sentinel-1 SAR images.
        Args: 
            manifest_path (str): Path to the dataset manifest CSV file.
            split (str): Dataset split - 'train', 'val', or 'test'.
            augment (bool): Whether to apply data augmentations.
        """
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.split = split
        self.augment = augment

        logger.info(f"[{split}] Total samples: {len(self.df)}")

        additional = {'image0': 'image', 'mask0': 'mask', 'mask1': 'mask'}

        if self.augment and split == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ], additional_targets=additional)
        else:
            self.transform = A.Compose([ToTensorV2()], additional_targets=additional)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
    
        try:
            # Load SAR image (3 channels: VV, VH, VV-VH)
            sar_img = load_and_stack_sar(row['s1_vv'], row['s1_vh'])  # Lee filtering done here
            sar_img = normalize_sar(sar_img)
    
            # Load RGB optical image (3 channels)
            opt_img = load_and_stack_optical(
                row['s2_b4_red'], row['s2_b3_green'], row['s2_b2_blue'])
            opt_img = normalize_optical(opt_img)
    
            # Load masks
            cloud_mask = load_mask(row['s2_cloudmask'])
            water_mask = load_mask(row['s1_vv'].replace('VV.tif', 'LabelWater.tif'))
    
            # Move to HWC for albumentations
            sar_img = np.moveaxis(sar_img, 0, -1)
            opt_img = np.moveaxis(opt_img, 0, -1)
    
            # Apply augmentations
            augmented = self.transform(
                image=sar_img,
                image0=opt_img,
                mask0=cloud_mask,
                mask1=water_mask
            )
    
            # Convert augmented outputs to tensors (assuming CHW from ToTensorV2)
            sar_tensor = augmented['image'].float()  # (3, H, W)
            opt_rgb = augmented['image0'].float()    # (3, H, W)
            water_mask = augmented['mask1'].float().unsqueeze(0)  # (1, H, W)
    
            # Ensure matching spatial dimensions and create 4-channel optical tensor
            if opt_rgb.size()[1:] != water_mask.size()[1:]:  # Check H, W dimensions
                raise ValueError(f"Spatial dimensions mismatch: opt_rgb {opt_rgb.size()}, water_mask {water_mask.size()}")
            opt_tensor = torch.cat((opt_rgb, water_mask), dim=0)  # (4, H, W)
    
            cloud_tensor = augmented['mask0'].float().unsqueeze(0)  # (1, H, W)
            water_tensor = water_mask  # (1, H, W)
    
            return sar_tensor, opt_tensor, cloud_tensor, water_tensor
    
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            return None


