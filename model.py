import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ImageEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.model.config.hidden_size, 512)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(512, 512)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        image_features = outputs.pooler_output  # [batch_size, hidden_size]
        projected = self.fc(self.gelu(self.projection(image_features)))
        return projected

class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.model.config.hidden_size, 512)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(512, 512)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.pooler_output
        projected = self.fc(self.gelu(self.projection(text_features)))
        return projected
