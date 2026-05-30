import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def scale_to_uint8(array: np.ndarray) -> np.ndarray:
    """Scales a numpy array to fit into uint8 range [0, 255]. Handles empty/flat arrays."""
    # Ensure array is not empty and has variance
    if array.size == 0 or np.all(array == array.flat[0]):
        return np.zeros(array.shape, dtype=np.uint8)

    p2, p98 = np.percentile(array, (2, 98))
    scaled_array = np.clip(array, p2, p98)

    # Avoid division by zero if the clipped array is flat
    min_val, max_val = np.min(scaled_array), np.max(scaled_array)
    if max_val == min_val:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled_array = (scaled_array - min_val) / (max_val - min_val)
    return (scaled_array * 255).astype(np.uint8)

def verify_geotiff(file_path: Path):
    """
    Opens, processes, and displays a GeoTIFF image for verification.
    Handles valid S1/S2 files and attempts to debug invalid ones.
    """
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        return

    try:
        with rasterio.open(file_path) as src:
            logging.info(f"Opening {file_path.name}...")
            logging.info(f"  - Bands: {src.count}")
            logging.info(f"  - CRS: {src.crs}")
            logging.info(f"  - Dimensions: {src.width}x{src.height}")

            if src.crs is None:
                logging.warning("File is NOT georeferenced. This strongly indicates a download error.")

            if src.count >= 5: # Expected Sentinel-2 L2A
                b04_red, b03_green, b02_blue = src.read(3), src.read(2), src.read(1)
                rgb_image = np.dstack((scale_to_uint8(b04_red), scale_to_uint8(b03_green), scale_to_uint8(b02_blue)))
                plt.figure(figsize=(10, 10))
                plt.title(f"Sentinel-2 (True Color)\n{file_path.name}")
                plt.imshow(rgb_image)
                plt.axis('off')

            elif src.count == 2: # Expected Sentinel-1 GRD
                vh_band = src.read(2)
                vh_display = scale_to_uint8(vh_band)
                plt.figure(figsize=(10, 10))
                plt.title(f"Sentinel-1 (VH Polarization)\n{file_path.name}")
                plt.imshow(vh_display, cmap='gray')
                plt.axis('off')

            # --- NEW DEBUGGING LOGIC ---
            else: # Handle unexpected band counts
                logging.warning(f"Unexpected band count ({src.count}). Displaying first band for debugging.")
                if src.count > 0:
                    band1 = src.read(1)
                    plt.figure(figsize=(10, 2)) # Adjust figure size to reflect weird dimensions
                    plt.title(f"DEBUG VIEW: Unexpected Data\n{file_path.name}")
                    plt.imshow(band1, cmap='viridis', aspect='auto')
                    plt.colorbar()
                    logging.info(f"Band 1 stats: min={np.min(band1)}, max={np.max(band1)}, mean={np.mean(band1)}")
                else:
                    logging.error("File has 0 bands and cannot be read.")

            plt.show()

    except Exception as e:
        logging.error(f"Could not process file {file_path}. Reason: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a downloaded GeoTIFF image.")
    parser.add_argument("filepath", type=Path, help="Path to the .tiff file to verify.")
    args = parser.parse_args()

    verify_geotiff(args.filepath)
