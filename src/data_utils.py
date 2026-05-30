import numpy as np
import rasterio
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
import logging
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def lee_filter(img, size):
    logger.debug("Applying Lee filter with window size=%s", size)
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2
    overall_variance = variance(img)
    img_weights = img_variance / (img_variance + overall_variance + 1e-8)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def load_image(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

def load_and_stack_sar(vv_path, vh_path):
    logger.debug("Loading SAR VV from %s and VH from %s", vv_path, vh_path)
    s1_vv = lee_filter(load_image(vv_path), size=5)
    s1_vh = lee_filter(load_image(vh_path), size=5)
    s1_diff = s1_vv - s1_vh
    return np.stack([s1_vv, s1_vh, s1_diff], axis=0)  # (3, H, W)

def load_and_stack_optical(r_path, g_path, b_path):
    logger.debug("Loading optical R=%s G=%s B=%s", r_path, g_path, b_path)
    s2_r = load_image(r_path)
    s2_g = load_image(g_path)
    s2_b = load_image(b_path)
    return np.stack([s2_r, s2_g, s2_b], axis=0)

def normalize_sar(sar_img):
    logger.debug("Normalizing SAR image to [-1, 1] range")
    norm_channels = []
    for i in range(sar_img.shape[0]):
        ch = sar_img[i]
        ch_min = ch.min()
        ch_max = ch.max()
        norm = 2 * (ch - ch_min) / (ch_max - ch_min + 1e-8) - 1
        norm_channels.append(norm)
    return np.stack(norm_channels, axis=0)

def normalize_optical(optical_img):
    logger.debug("Normalizing optical image to [-1, 1] collectively")
    min_val = optical_img.min()
    max_val = optical_img.max()
    return 2 * (optical_img - min_val) / (max_val - min_val + 1e-8) - 1

def load_mask(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.uint8)
    return (mask > 0).astype(np.float32)

def load_cloud_mask(mask_path):
    """Alias for load_mask — loads a binary cloud mask from a GeoTIFF."""
    return load_mask(mask_path)

def apply_speckle_filter(img, size=5):
    """Apply a Lee speckle filter to a single-channel SAR image."""
    logger.debug("Applying speckle (Lee) filter with window size=%s", size)
    return lee_filter(img, size=size)

def get_cloud_coverage(cloud_mask):
    """Return the fraction of pixels flagged as cloud (0.0 – 1.0)."""
    return float(cloud_mask.mean())