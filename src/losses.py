import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
import logging
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

#1. Adversarial Loss
class LSGANLoss(nn.Module):
    def __init__(self, target_real_label=0.9, target_fake_label=0.0):
        '''
        Adversarial loss for the generator.
        Args:
            target_real_label (float): Target label for real images.
            target_fake_label (float): Target label for fake images.
        '''
        super(LSGANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def forward(self, prediction, target_is_real):
        logger.debug("LSGANLoss called, target_is_real=%s", target_is_real)
        target_tensor = self.real_label if target_is_real else self.fake_label
        return self.loss(prediction, target_tensor.expand_as(prediction))

# 2. Perceptual Loss - VGG19 Feature Loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.loss = nn.L1Loss()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, gen_img, real_img, cloud_mask):
        logger.debug("PerceptualLoss forward with gen_img shape %s", tuple(gen_img.shape))
        gen_img = (gen_img + 1) / 2
        real_img = (real_img + 1) / 2
        gen_img = (gen_img - self.mean) / self.std
        real_img = (real_img - self.mean) / self.std

        gen_features = self.vgg(gen_img)
        real_features = self.vgg(real_img)
        
        mask = F.interpolate(cloud_mask, size=gen_features.shape[2:], mode='bilinear', align_corners=False)
        mask = (1 - mask).expand_as(gen_features)
        masked_gen_features = gen_features * mask
        masked_real_features = real_features * mask
        return self.loss(masked_gen_features, masked_real_features)

# 3. Custom Speckle Preservation Loss
class SpecklePreservationLoss(nn.Module):
    def __init__(self):
        super(SpecklePreservationLoss, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.loss = nn.L1Loss()

    def get_gradient_magnitude(self, img):
        if img.shape[1] == 3:
            img_gray = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        elif img.shape[1] == 2:
            img_gray = img[:, 0:1, :, :]
        else:
            img_gray = img
        grad_x = F.conv2d(img_gray, self.sobel_x, padding='same')
        grad_y = F.conv2d(img_gray, self.sobel_y, padding='same')
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

    def forward(self, gen_img, sar_img, cloud_mask):
        logger.debug("SpecklePreservationLoss forward")
        with torch.cuda.amp.autocast():
            grad_gen = self.get_gradient_magnitude(gen_img)
            grad_sar = self.get_gradient_magnitude(sar_img)
        weight_map = torch.exp(-grad_sar)
        
        mask = (1 - cloud_mask).expand_as(grad_gen)
        weighted_grad_gen = weight_map * grad_gen * mask
        return self.loss(weighted_grad_gen, torch.zeros_like(weighted_grad_gen))

# 4. WaterIndexConsistencyLoss
class WaterIndexConsistencyLoss(nn.Module):
    def __init__(self, ndwi_weight=0.5):
        super(WaterIndexConsistencyLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        # ndwi_weight is kept for backward compatibility but won't be used
        self.ndwi_weight = ndwi_weight

    def forward(self, gen_img, water_mask):
        logger.debug("WaterIndexConsistencyLoss forward with gen_img shape %s", gen_img.shape)

        # The 4th channel is the water mask logits
        water_logits = gen_img[:, 3:4, :, :]
        
        # We drop the NDWI proxy loss because calculating (G-R)/(G+R) was causing
        # mode collapse (forcing the model to output purely Red and Green).
        total_loss = self.bce_loss(water_logits, water_mask)
        return total_loss