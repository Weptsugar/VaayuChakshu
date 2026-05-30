import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import yaml
os.environ["WANDB_MODE"] = "offline"
from argparse import ArgumentParser, Namespace
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import logging
from logging_utils import setup_logging

from lightning_module import SAR2OpticalGAN
from datamodule import SARDataModule

setup_logging()
logger = logging.getLogger(__name__)

def main(hparams):
    logger.info("Initialising W&B logger for project '%s' run '%s'", hparams.project_name, hparams.run_name)
    wandb_logger = WandbLogger(
        project=hparams.project_name,
        name=hparams.run_name,
    )

    logger.debug("Logging hyperparameters to W&B")
    wandb_logger.log_hyperparams(hparams)

    logger.info("Creating model checkpoint callback")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{hparams.run_name}",
        filename='{epoch:02d}-{val_psnr:.4f}',
        save_top_k=3,
        verbose=True,
        monitor='val/psnr',
        every_n_epochs=1,
        mode='max',
        save_last=True
    )
    early_stopping = EarlyStopping(monitor='train_loss/generator_total', mode='min', patience=10, verbose=True)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger.info("Instantiating GAN model and datamodule")
    model = SAR2OpticalGAN(hparams)
    datamodule = SARDataModule(hparams)

    logger.info("Initialising Trainer with max_epochs=%s", hparams.max_epochs)
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        logger=wandb_logger,
        check_val_every_n_epoch=5,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        accelerator=hparams.accelerator,
        devices=1,
        strategy="auto",
        precision=hparams.precision,
        log_every_n_steps=hparams.log_every_n_steps,
    )
    logger.info("Starting training...")
    trainer.fit(model, datamodule)
    logger.info("Training complete")

    logger.debug("Closing W&B experiment")
    wandb_logger.experiment.finish()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    logger.debug("Loading YAML configuration from %s", args.config)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    hparams = Namespace(**config)
    logger.debug("Parsed hyperparameters: %s", hparams)
    try:
        main(hparams)
    except Exception as e:
        logger.exception("Unhandled exception during training: %s", str(e))
        raise