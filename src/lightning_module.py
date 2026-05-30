import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from argparse import Namespace
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.optim.swa_utils import AveragedModel
from copy import deepcopy
import torch.distributed as dist

from model import UNetGenerator, PatchGANDiscriminator
from losses import LSGANLoss, PerceptualLoss, SpecklePreservationLoss, WaterIndexConsistencyLoss

class SAR2OpticalGAN(pl.LightningModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False

        self.generator = UNetGenerator(in_channels=3, out_channels=4)
        self.discriminator = PatchGANDiscriminator(in_channels=7)

        self.generator_ema = AveragedModel(
            deepcopy(self.generator),
            avg_fn=torch.optim.swa_utils.get_ema_avg_fn(self.hparams.ema_decay)
        )

        self.adv_loss = LSGANLoss(target_real_label=0.9, target_fake_label=0.05)
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.speckle_loss = SpecklePreservationLoss()
        self.water_loss = WaterIndexConsistencyLoss()
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def on_train_start(self):
        self.generator_ema.to(self.device)

    def forward(self, sar_img):
        return self.generator_ema(sar_img)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        sar_img, opt_img, cloud_mask, water_mask = batch
        clear_pixels_mask = 1 - cloud_mask

        if self.trainer.world_size > 1:
            dist.barrier()

        try:
            with torch.cuda.amp.autocast(enabled=False):
                generated_opt = self.generator(sar_img)

                if torch.isnan(generated_opt).any():
                    raise ValueError("NaN in generated output")

                pred_fake_for_g = self.discriminator(torch.cat((sar_img, generated_opt), dim=1))

                loss_g_adv = self.hparams.lambda_adv * self.adv_loss(pred_fake_for_g, True)
                masked_fake_rgb = generated_opt[:, :3, :, :] * clear_pixels_mask
                masked_real_rgb = opt_img[:, :3, :, :] * clear_pixels_mask

                loss_g_l1 = self.hparams.lambda_l1 * self.l1_loss(masked_fake_rgb, masked_real_rgb)
                loss_g_perc = self.hparams.lambda_perc * self.perceptual_loss(masked_fake_rgb, masked_real_rgb, cloud_mask)
                loss_g_speckle = self.hparams.lambda_speckle * self.speckle_loss(generated_opt[:, :3, :, :], sar_img, cloud_mask)
                loss_g_water = self.hparams.lambda_water * self.water_loss(generated_opt, water_mask)

                loss_g = loss_g_adv + loss_g_l1 + loss_g_perc + loss_g_speckle + loss_g_water

                if torch.isnan(loss_g):
                    raise ValueError("NaN in generator total loss")

            opt_g.zero_grad()
            self.manual_backward(loss_g)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            opt_g.step()

            self.generator_ema.update_parameters(self.generator)

            loss_d = None
            if (batch_idx + 1) % self.hparams.discriminator_update_freq == 0:
                with torch.cuda.amp.autocast(enabled=False):
                    generated_opt_detached = generated_opt.detach()
                    pred_real = self.discriminator(torch.cat((sar_img, opt_img), dim=1))
                    loss_d_real = self.adv_loss(pred_real, True)
                    pred_fake = self.discriminator(torch.cat((sar_img, generated_opt_detached), dim=1))
                    loss_d_fake = self.adv_loss(pred_fake, False)
                    loss_d = (loss_d_real + loss_d_fake) * 0.5

                    if torch.isnan(loss_d):
                        raise ValueError("NaN in discriminator loss")

                opt_d.zero_grad()
                self.manual_backward(loss_d)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                opt_d.step()

            if self.trainer.is_global_zero:
                self.log("train/g_loss", loss_g, prog_bar=True, logger=False)
                log_dict = {
                    'train_loss/generator_total': loss_g,
                    'train_loss/g_adversarial': loss_g_adv,
                    'train_loss/g_l1': loss_g_l1,
                    'train_loss/g_perceptual': loss_g_perc,
                    'train_loss/g_speckle': loss_g_speckle,
                    'train_loss/g_water_consistency': loss_g_water
                }
                if loss_d is not None:
                    log_dict['train/d_loss'] = loss_d
                    log_dict['train_loss/discriminator'] = loss_d
                self.log_dict(log_dict, logger=True)

            if self.current_epoch % 5 == 0 and batch_idx == 0 and self.logger and self.logger.experiment:
                num_samples = min(10, sar_img.size(0))
                images = []
                for i in range(num_samples):
                    sar = sar_img[i]
                    opt = opt_img[i, :3, :, :]
                    cloud = cloud_mask[i]
                    water = water_mask[i]
                    gen = generated_opt[i]
    
                    sar_vv = sar[0].unsqueeze(0).repeat(3, 1, 1)
                    cloud_3c = cloud.repeat(3, 1, 1)
                    water_3c = water.repeat(3, 1, 1)
                    gen_rgb = gen[:3]
    
                    image = torch.cat([sar_vv, opt, cloud_3c, water_3c, gen_rgb], dim=2)
                    images.append(wandb.Image(image, caption=f"Sample {i} | SAR | OPT | Cloud | Water | Gen | Epoch {self.current_epoch}"))
    
                self.logger.experiment.log({
                    "train_samples": images,
                    "epoch": self.current_epoch
                })
                
                # Checkpoint syncing is handled by ModelCheckpoint callback (local run)
            
        
        except Exception as e:
            self.logger.experiment.log({"error": str(e), "epoch": self.current_epoch})
            raise e

        return loss_g

    def validation_step(self, batch, batch_idx):
        sar_img, opt_img, cloud_mask, water_mask = batch
        generated_opt = self(sar_img)

        val_l1_loss = self.l1_loss(generated_opt[:, :3, :, :], opt_img[:, :3, :, :])

        opt_img_01 = (opt_img[:, :3, :, :] + 1) / 2
        generated_opt_01 = (generated_opt[:, :3, :, :] + 1) / 2

        self.val_psnr(generated_opt_01, opt_img_01)
        self.val_ssim(generated_opt_01, opt_img_01)

        self.log_dict(
            {'val/psnr': self.val_psnr, 'val/ssim': self.val_ssim, 'val/l1_loss': val_l1_loss},
            on_epoch=True, prog_bar=True, sync_dist=True
        )
        if batch_idx == 0 and self.trainer.is_global_zero and self.logger and self.logger.experiment:
            num_samples = min(10, sar_img.size(0))
            images = []
            for i in range(num_samples):
                sar = sar_img[i]           # (3, 512, 512)
                opt = opt_img[i, :3, :, :] # Use only RGB channels (3, 512, 512)
                cloud = cloud_mask[i]      # (1, 512, 512)
                water = water_mask[i]      # (1, 512, 512)
                gen = generated_opt[i]     # (4, 512, 512)
    
                sar_vv = sar[0].unsqueeze(0).repeat(3, 1, 1)
                cloud_3c = cloud.repeat(3, 1, 1)
                water_3c = water.repeat(3, 1, 1)
                gen_rgb = gen[:3]
                image = torch.cat([sar_vv, opt, cloud_3c, water_3c, gen_rgb], dim=2)
                images.append(wandb.Image(image, caption=f"Sample {i} | SAR | OPT | Cloud | Water | Gen"))
    
            self.logger.experiment.log({
                "val_samples": images,
                "epoch": self.current_epoch
            })

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        sch_g, sch_d = self.lr_schedulers()
        psnr_metric = self.trainer.callback_metrics.get("val/psnr")
        d_loss_metric = self.trainer.callback_metrics.get("train/d_loss_epoch")
        if psnr_metric is not None:
            sch_g.step(psnr_metric)
        if d_loss_metric is not None:
            sch_d.step(d_loss_metric)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=1e-5, betas=(0.5, 0.999))
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))

        scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_g, mode='max', factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience
        )
        scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_d, mode='min', factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience
        )

        return (
            {"optimizer": opt_g, "lr_scheduler": {"scheduler": scheduler_g, "monitor": "val/psnr"}},
            {"optimizer": opt_d, "lr_scheduler": {"scheduler": scheduler_d, "monitor": "train/d_loss_epoch"}}
        )

    def on_train_batch_start(self, batch, batch_idx):
        batch = [t.to(self.device) for t in batch]
