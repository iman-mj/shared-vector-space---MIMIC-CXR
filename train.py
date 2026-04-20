import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, AutoTokenizer
from model import ImageEncoder, TextEncoder
from clip import MIMIC_CLIP, get_lr_scheduler
from test import calculate_metrics, plot_loss_curves
from dataset import MIMICDataset

PATH_CONFIG = {
    "train_json": "/home/mimic-cxr/dataset/reports/train.jsonl",
    "valid_json": "/home/mimic-cxr/dataset/reports/valid.jsonl",
    "test_json": "/home/mimic-cxr/dataset/reports/test.jsonl",
    "checkpoint_dir": "./checkpoints"
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_ds = MIMICDataset(PATH_CONFIG["train_json"], processor, tokenizer)
    valid_ds = MIMICDataset(PATH_CONFIG["valid_json"], processor, tokenizer)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=32, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True
    )
    
    valid_loader = DataLoader(valid_ds, batch_size=32, num_workers=4)

    model = MIMIC_CLIP().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
    scheduler = get_lr_scheduler(optimizer, len(train_loader) * 10)

    train_losses, valid_losses = [], []
    
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            loss, logits = model(pixel_values, input_ids, attention_mask)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        print(f"Epoch {epoch} finished. Train Loss: {avg_train_loss:.4f}")
        
    plot_loss_curves(train_losses, valid_losses)

if __name__ == "__main__":
    main()
