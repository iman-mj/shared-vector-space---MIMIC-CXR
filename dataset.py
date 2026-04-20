import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class MIMICDataset(Dataset):
    def __init__(self, jsonl_file, processor, tokenizer, max_length=120):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        img_path = item['img']
        image = Image.open(img_path).convert("RGB")
        
        image_input = self.processor(
            images=image, 
            return_tensors="pt"
        )

        report_text = item['text']
        
        text_input = self.tokenizer(
            report_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'pixel_values': image_input['pixel_values'].squeeze(0), 
            'input_ids': text_input['input_ids'].squeeze(0),       
            'attention_mask': text_input['attention_mask'].squeeze(0), 
            'label': item['label']
        }
