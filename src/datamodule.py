from torch.utils.data._utils.collate import default_collate
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import FloodDataset
import logging
from logging_utils import setup_logging
import torch

setup_logging()
logger = logging.getLogger(__name__)

def collate_fn(batch):
    '''
    Custom collate function to filter out None values.
    Returns zero tensors if batch is empty to avoid training crashes.
    '''
    filtered = list(filter(lambda x: x is not None, batch))
    if len(filtered) != len(batch):
        logger.warning("Batch contained %s invalid samples; they were filtered out", len(batch) - len(filtered))
    if not filtered:
        h, w = 512, 512
        return (torch.zeros(4, 3, h, w),)
    return default_collate(filtered)

class SARDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        logger.info("Creating SARDataModule")
        super().__init__()
        self.save_hyperparameters(hparams)

    def setup(self, stage=None):
        logger.debug("Setting up datasets for stage=%s", stage)
        '''
        Called on every GPU/TPU in distributed training
        '''
        if stage == 'fit' or stage is None:
            logger.info("Instantiating FloodDataset for training and validation")
            self.train_dataset = FloodDataset(
                manifest_path=self.hparams.manifest_path,
                split='train',
                augment=True,
            )
            self.val_dataset = FloodDataset(
                manifest_path=self.hparams.manifest_path,
                split='val',
                augment=False,
            )

    def train_dataloader(self):
        logger.debug("Creating training DataLoader with batch_size=%s", self.hparams.batch_size)
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.get('num_workers', 4),
            pin_memory=self.hparams.get('pin_memory', True),
            persistent_workers=self.hparams.get('persistent_workers', True),
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        logger.debug("Creating validation DataLoader with batch_size=%s", self.hparams.batch_size)
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.get('num_workers', 4),
            pin_memory=self.hparams.get('pin_memory', True),
            persistent_workers=self.hparams.get('persistent_workers', True),
            collate_fn=collate_fn,
        )