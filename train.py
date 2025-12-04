"""
Training and evaluation script for LSTM sentiment classifier.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import numpy as np

from data import load_imdb_data
from model import LSTMSentimentClassifier, count_parameters
from utils import (
    calculate_metrics, 
    get_confusion_matrix, 
    save_model, 
    load_model,
    EarlyStopping,
    print_metrics,
    print_confusion_matrix
)


# Hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MAX_VOCAB_SIZE = 25000
MAX_LENGTH = 256
NUM_EPOCHS = 10
PATIENCE = 3


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Average loss, predictions, labels
    """
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training")
    for texts, labels in progress_bar:
        texts, labels = texts.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        predictions = (outputs >= 0.5).long()
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, np.array(all_predictions), np.array(all_labels)


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model on validation or test set.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Average loss, predictions, labels
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in tqdm(data_loader, desc="Evaluating"):
            texts, labels = texts.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            predictions = (outputs >= 0.5).long()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss, np.array(all_predictions), np.array(all_labels)


def train_model(train_loader, val_loader, vocab, device, num_epochs=NUM_EPOCHS):
    """
    Train the sentiment classifier.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        vocab: Vocabulary object
        device: Device to train on
        num_epochs: Number of epochs to train
        
    Returns:
        Trained model
    """
    # Initialize model
    model = LSTMSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE, mode='max')
    
    best_val_f1 = 0.0
    
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_metrics = calculate_metrics(train_preds, train_labels)
        
        # Validate
        val_loss, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        val_metrics = calculate_metrics(val_preds, val_labels)
        
        # Print results
        print(f"\nTrain Loss: {train_loss:.4f}")
        print_metrics(train_metrics, "Train")
        
        print(f"\nValidation Loss: {val_loss:.4f}")
        print_metrics(val_metrics, "Validation")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_model(model, vocab)
            print(f"\n✓ Best model saved (F1: {best_val_f1:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics['f1']):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}\n")
    
    return model


def test_model(test_loader, device):
    """
    Test the trained model.
    
    Args:
        test_loader: Test data loader
        device: Device to test on
    """
    print(f"\n{'='*60}")
    print("Testing Model")
    print(f"{'='*60}\n")
    
    # Load best model
    model, vocab = load_model(LSTMSentimentClassifier, device=device)
    
    # Evaluate
    criterion = nn.BCELoss()
    test_loss, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    test_metrics = calculate_metrics(test_preds, test_labels)
    
    # Print results
    print(f"Test Loss: {test_loss:.4f}")
    print_metrics(test_metrics, "Test")
    
    # Confusion matrix
    cm = get_confusion_matrix(test_preds, test_labels)
    print_confusion_matrix(cm)
    
    # Print some example predictions
    print(f"\n{'='*60}")
    print("Sample Predictions")
    print(f"{'='*60}\n")
    
    model.eval()
    with torch.no_grad():
        for i, (texts, labels) in enumerate(test_loader):
            if i >= 1:  # Only show first batch
                break
            
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            predictions = (outputs >= 0.5).long()
            
            # Show first 5 examples
            for j in range(min(5, len(texts))):
                text_decoded = vocab.decode(texts[j].cpu().numpy())
                # Truncate long texts
                if len(text_decoded) > 100:
                    text_decoded = text_decoded[:100] + "..."
                
                print(f"Example {j+1}:")
                print(f"  Text: {text_decoded}")
                print(f"  True Label: {'Positive' if labels[j].item() == 1 else 'Negative'}")
                print(f"  Predicted: {'Positive' if predictions[j].item() == 1 else 'Negative'}")
                print(f"  Probability: {outputs[j].item():.4f}")
                print()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train LSTM sentiment classifier')
    parser.add_argument('--evaluate', action='store_true', help='Only evaluate the model')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading and preprocessing data...")
    train_loader, val_loader, test_loader, vocab = load_imdb_data(
        train_split=0.7,
        val_split=0.15,
        max_vocab_size=MAX_VOCAB_SIZE,
        max_length=MAX_LENGTH,
        batch_size=args.batch_size
    )
    
    if args.evaluate:
        # Only evaluate
        test_model(test_loader, device)
    else:
        # Train and evaluate
        model = train_model(train_loader, val_loader, vocab, device, num_epochs=args.epochs)
        test_model(test_loader, device)


if __name__ == "__main__":
    main()