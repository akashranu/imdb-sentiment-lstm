"""
Data loading and preprocessing for IMDB sentiment analysis.
"""

import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset
import numpy as np


class Vocabulary:
    """Vocabulary class for mapping words to indices."""
    
    def __init__(self, max_vocab_size=25000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_counts = Counter()
        
    def build_vocabulary(self, texts):
        """Build vocabulary from a list of texts."""
        # Count word frequencies
        for text in texts:
            tokens = self._tokenize(text)
            self.word_counts.update(tokens)
        
        # Get most common words
        most_common = self.word_counts.most_common(self.max_vocab_size - 2)
        
        # Add to vocabulary
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        print(f"Vocabulary built with {len(self.word2idx)} tokens")
        
    def _tokenize(self, text):
        """Tokenize text into words."""
        # Lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = text.split()
        return tokens
    
    def encode(self, text):
        """Convert text to list of indices."""
        tokens = self._tokenize(text)
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
    
    def decode(self, indices):
        """Convert list of indices back to text."""
        return ' '.join([self.idx2word.get(idx, "<UNK>") for idx in indices])
    
    def __len__(self):
        return len(self.word2idx)


class IMDBDataset(Dataset):
    """PyTorch Dataset for IMDB reviews."""
    
    def __init__(self, texts, labels, vocab, max_length=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        encoded = self.vocab.encode(text)
        
        # Pad or truncate
        if len(encoded) < self.max_length:
            encoded = encoded + [0] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
            
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.float)


def load_imdb_data(train_split=0.7, val_split=0.15, max_vocab_size=25000, max_length=256, batch_size=32):
    """
    Load and preprocess IMDB dataset.
    
    Args:
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        max_vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length
        batch_size: Batch size for DataLoader
        
    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    
    # Get train and test data
    train_data = dataset['train']
    test_data = dataset['test']
    
    # Split training data into train and validation
    train_size = int(len(train_data) * train_split / (train_split + val_split))
    
    train_texts = train_data['text'][:train_size]
    train_labels = train_data['label'][:train_size]
    
    val_texts = train_data['text'][train_size:]
    val_labels = train_data['label'][train_size:]
    
    test_texts = test_data['text']
    test_labels = test_data['label']
    
    print(f"Train size: {len(train_texts)}")
    print(f"Validation size: {len(val_texts)}")
    print(f"Test size: {len(test_texts)}")
    
    # Build vocabulary from training data
    print("\nBuilding vocabulary...")
    vocab = Vocabulary(max_vocab_size=max_vocab_size)
    vocab.build_vocabulary(train_texts)
    
    # Create datasets
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, max_length)
    val_dataset = IMDBDataset(val_texts, val_labels, vocab, max_length)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader, vocab = load_imdb_data()
    
    # Print sample batch
    for texts, labels in train_loader:
        print(f"\nBatch shape: {texts.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sample text (encoded): {texts[0][:20]}")
        print(f"Sample label: {labels[0]}")
        break