import os
import csv
import random
import logging
import re
from glob import glob

from src.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def create_dataset_manifest(data_root, output_csv, test_size=0.15, random_state=42):
    """
    Parses the C2S-MS Floods dataset structure, correctly pairs S1 and S2 chips based on the
    actual file naming convention, and creates a manifest file with train/val splits.
    Only uses Python standard libraries to avoid external dependency requirements.

    Args:
        data_root (str): Root directory where the event folders (UUIDs) are located.
        output_csv (str): Path to save the output CSV manifest file.
        test_size (float): The proportion of the dataset to allocate to the validation set.
        random_state (int): Seed for the random split for reproducibility.
    """
    try:
        # Search for event directories
        event_dirs = [d for d in glob(os.path.join(data_root, '*')) if os.path.isdir(d)]
        if not event_dirs:
            logger.error(f"No event directories found in {data_root}. Please check the path.")
            return
        logger.info(f"Found {len(event_dirs)} event directories. Starting scan...")

        records = []
        # regex to extract the coordinate ID (e.g. - '01845-00514')
        id_parser = re.compile(r'(\d{5}-\d{5})$')
        
        for event_dir in event_dirs:
            event_name = os.path.basename(event_dir)
            s1_chip_dirs = glob(os.path.join(event_dir, 's1', '*'))
            s2_chip_dirs = glob(os.path.join(event_dir, 's2', '*'))

            # Create dictionaries mapping the unique ID to the full path
            s1_map = {}
            for p in s1_chip_dirs:
                match = id_parser.search(os.path.basename(p))
                if match:
                    s1_map[match.group(1)] = p
            
            s2_map = {}
            for p in s2_chip_dirs:
                match = id_parser.search(os.path.basename(p))
                if match:
                    s2_map[match.group(1)] = p

            common_ids = s1_map.keys() & s2_map.keys()

            if not common_ids:
                logger.warning(f"No matching chip IDs found for event {event_name}. Skipping event.")
                continue

            logger.info(f"Processing event {event_name}: Found {len(common_ids)} pairs.")

            for chip_id in common_ids:
                s1_chip_dir = s1_map[chip_id]
                s2_chip_dir = s2_map[chip_id]
                
                s1_vv_path = os.path.join(s1_chip_dir, 'VV.tif') 
                s1_vh_path = os.path.join(s1_chip_dir, 'VH.tif') 
                
                s2_b4_path = os.path.join(s2_chip_dir, 'B4.tif') 
                s2_b3_path = os.path.join(s2_chip_dir, 'B3.tif') 
                s2_b2_path = os.path.join(s2_chip_dir, 'B2.tif')
                
                s2_cloudmask_path = os.path.join(s2_chip_dir, 'LabelCloud.tif')

                required_files = [s1_vv_path, s1_vh_path, s2_b4_path, s2_b3_path, s2_b2_path, s2_cloudmask_path]
                if all(os.path.exists(p) for p in required_files):
                    records.append({
                        's1_vv': s1_vv_path.replace(os.sep, '/'),
                        's1_vh': s1_vh_path.replace(os.sep, '/'),
                        's2_b4_red': s2_b4_path.replace(os.sep, '/'),
                        's2_b3_green': s2_b3_path.replace(os.sep, '/'),
                        's2_b2_blue': s2_b2_path.replace(os.sep, '/'),
                        's2_cloudmask': s2_cloudmask_path.replace(os.sep, '/'),
                        'event': event_name,
                        's1_chip_id': os.path.basename(s1_chip_dir),
                        's2_chip_id': os.path.basename(s2_chip_dir)
                    })
                else:
                    logger.warning(f"Missing one or more files in chip pair: "
                                   f"S1: {os.path.basename(s1_chip_dir)}, S2: {os.path.basename(s2_chip_dir)}. Skipping.")

        if not records:
            logger.error("No valid records were created. This can happen if files are still missing or paths are incorrect.")
            return

        logger.info(f"Successfully processed {len(records)} complete chip pairs.")

        # Train/validation split using random (reproducible with seed)
        random.seed(random_state)
        random.shuffle(records)
        
        val_count = int(len(records) * test_size)
        
        for i, record in enumerate(records):
            if i < val_count:
                record['split'] = 'val'
            else:
                record['split'] = 'train'

        # Write to CSV
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        headers = ['s1_vv', 's1_vh', 's2_b4_red', 's2_b3_green', 's2_b2_blue', 's2_cloudmask', 'event', 's1_chip_id', 's2_chip_id', 'split']
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for record in records:
                writer.writerow(record)
                
        logger.info(f"Manifest file created successfully at: {output_csv}")
        
        train_count = len(records) - val_count
        logger.info(f"Dataset summary: train={train_count}, val={val_count}")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    DATASET_ROOT = os.path.join("data", "raw", "data", "c2s_ms_floods", "chips")
    OUTPUT_MANIFEST = os.path.join("data", "processed", "data_manifest.csv")

    create_dataset_manifest(DATASET_ROOT, OUTPUT_MANIFEST)