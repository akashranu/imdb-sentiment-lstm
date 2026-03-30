import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import json


def calculate_metrics(predictions, labels):
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def get_confusion_matrix(predictions, labels):
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    return confusion_matrix(labels, predictions)


def save_model(model, vocab, save_dir='saved_models', model_name='best_model.pt'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state
    model_path = os.path.join(save_dir, model_name)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': len(vocab),
        'embedding_dim': model.embedding.embedding_dim,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
    }, model_path)
    
    # Save vocabulary
    vocab_path = os.path.join(save_dir, 'vocab.json')
    vocab_data = {
        'word2idx': vocab.word2idx,
        'idx2word': {str(k): v for k, v in vocab.idx2word.items()},
        'max_vocab_size': vocab.max_vocab_size
    }
    with open(vocab_path, 'w') as f:
        json.dump(vocab_data, f)
    
    print(f"Model saved to {model_path}")
    print(f"Vocabulary saved to {vocab_path}")


def load_model(model_class, save_dir='saved_models', model_name='best_model.pt', device='cpu'):
    from data import Vocabulary
    
    # Load model checkpoint
    model_path = os.path.join(save_dir, model_name)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = model_class(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load vocabulary
    vocab_path = os.path.join(save_dir, 'vocab.json')
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    vocab = Vocabulary(max_vocab_size=vocab_data['max_vocab_size'])
    vocab.word2idx = vocab_data['word2idx']
    vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
    
    print(f"Model loaded from {model_path}")
    print(f"Vocabulary loaded from {vocab_path}")
    
    return model, vocab


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def print_metrics(metrics, prefix=""):
    print(f"{prefix} Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")


def print_confusion_matrix(cm):
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Neg    Pos")
    print(f"Actual Neg  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       Pos  {cm[1][0]:5d}  {cm[1][1]:5d}")


if __name__ == "__main__":
    # Test metrics calculation
    predictions = np.array([0, 1, 1, 0, 1, 1, 0, 0])
    labels = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    
    metrics = calculate_metrics(predictions, labels)
    print_metrics(metrics, "Test")
    
    cm = get_confusion_matrix(predictions, labels)
    print_confusion_matrix(cm)
