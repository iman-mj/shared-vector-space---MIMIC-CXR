import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def calculate_metrics(logits):
    logits = logits.cpu().numpy()
    labels = np.arange(len(logits))
    
    ranks = []
    for i in range(len(logits)):
        sorted_indices = np.argsort(logits[i])[::-1]
        rank = np.where(sorted_indices == labels[i])[0][0]
        ranks.append(rank)
    
    ranks = np.array(ranks)
    recall_1 = (ranks < 1).mean() * 100
    recall_5 = (ranks < 5).mean() * 100
    mrr = (1 / (ranks + 1)).mean()
    
    return recall_1, recall_5, mrr

def plot_similarity_heatmap(logits, n_samples=10):
    plt.figure(figsize=(10, 8))
    subset = logits[:n_samples, :n_samples].cpu().numpy()
    sns.heatmap(subset, annot=True, cmap='Blues', fmt='.2f')
    plt.title(f"Image-Text Similarity Matrix (First {n_samples} samples)")
    plt.xlabel("Text Embeddings")
    plt.ylabel("Image Embeddings")
    plt.savefig("similarity_heatmap.png")
    plt.show()

def plot_loss_curves(train_losses, valid_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.title("Fine-tuning CLIP on MIMIC-CXR")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve.png")
    plt.show()

if __name__ == "__main__":
    mock_logits = torch.randn(32, 32)
    r1, r5, mrr = calculate_metrics(mock_logits)
    print(f"Results -> Recall@1: {r1:.2f}%, Recall@5: {r5:.2f}%, MRR: {mrr:.4f}")
