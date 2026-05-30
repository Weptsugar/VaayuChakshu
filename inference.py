import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import argparse
import torch
import numpy as np
import rasterio
from PIL import Image

from logging_utils import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

from lightning_module import SAR2OpticalGAN
from data_utils import load_and_stack_sar, normalize_sar

def run_inference(checkpoint_path, s1_vv_path, s1_vh_path, output_dir="outputs/results"):
    """
    Runs inference on a Sentinel-1 SAR pair using a trained Flood-GAN checkpoint.
    Saves the generated optical RGB image and the flood water mask.
    """
    logger.info("Loading model from checkpoint: %s", checkpoint_path)
    # Load model (compiled weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAR2OpticalGAN.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    
    logger.info("Loading and preprocessing input SAR image...")
    # Load and stack VV and VH channels (applies Lee filter)
    sar_stacked = load_and_stack_sar(s1_vv_path, s1_vh_path)
    sar_normalized = normalize_sar(sar_stacked)
    
    # Add batch dimension and convert to torch tensor
    sar_tensor = torch.from_numpy(sar_normalized).float().unsqueeze(0).to(device) # (1, 3, H, W)
    
    logger.info("Running forward pass through Generator...")
    with torch.no_grad():
        # Generator outputs a 4-channel tensor (RGB + Water mask logits)
        generated_output = model(sar_tensor)
        
    # Remove batch dimension
    generated_output = generated_output.squeeze(0).cpu() # (4, H, W)
    
    # 1. Post-process Generated RGB Image
    rgb_output = generated_output[:3, :, :] # (3, H, W)
    # Convert from [-1, 1] scale back to [0, 255] uint8
    rgb_output = ((rgb_output + 1) / 2.0 * 255.0).clamp(0, 255).numpy().astype(np.uint8)
    rgb_image = np.moveaxis(rgb_output, 0, -1) # (H, W, 3)
    
    # 2. Post-process Generated Flood Water Mask
    water_logits = generated_output[3, :, :] # (H, W)
    # Apply sigmoid to get probabilities, then threshold at 0.5 for binary mask
    water_probs = torch.sigmoid(water_logits).numpy()
    water_mask = (water_probs > 0.5).astype(np.uint8) * 255 # (H, W)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    s1_id = os.path.basename(os.path.dirname(s1_vv_path))
    rgb_save_path = os.path.join(output_dir, f"{s1_id}_gen_optical.png")
    mask_save_path = os.path.join(output_dir, f"{s1_id}_gen_flood_mask.png")
    
    Image.fromarray(rgb_image).save(rgb_save_path)
    Image.fromarray(water_mask).save(mask_save_path)
    
    logger.info("Inference complete!")
    logger.info("Saved generated optical image to: %s", rgb_save_path)
    logger.info("Saved generated flood mask to: %s", mask_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flood-GAN Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained PyTorch Lightning checkpoint file (.ckpt)")
    parser.add_argument("--s1_vv", type=str, required=True, help="Path to the input Sentinel-1 VV .tif file")
    parser.add_argument("--s1_vh", type=str, required=True, help="Path to the input Sentinel-1 VH .tif file")
    parser.add_argument("--output_dir", type=str, default="outputs/results", help="Directory to save the generated results")
    
    args = parser.parse_args()
    
    try:
        run_inference(args.checkpoint, args.s1_vv, args.s1_vh, args.output_dir)
    except Exception as e:
        logger.exception("Error during inference execution: %s", str(e))
        raise
