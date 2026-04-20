import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ImageEncoder, TextEncoder

class MIMIC_CLIP(nn.Module):
    def __init__(self, image_model="google/vit-base-patch16-224", text_model="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.image_encoder = ImageEncoder(image_model)
        self.text_encoder = TextEncoder(text_model)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, pixel_values, input_ids, attention_mask):
        image_features = self.image_encoder(pixel_values)
        text_features = self.text_encoder(input_ids, attention_mask)

        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        logits = (image_features @ text_features.T) / torch.exp(self.temperature)
        
        labels = torch.arange(logits.size(0)).to(logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        return (loss_i + loss_t) / 2, logits

def get_lr_scheduler(optimizer, total_steps):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
