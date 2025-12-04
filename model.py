"""
LSTM-based sentiment analysis model.
"""

import torch
import torch.nn as nn


class LSTMSentimentClassifier(nn.Module):
    """
    LSTM-based sentiment classifier.
    
    Architecture:
        Embedding -> LSTM -> Fully Connected -> Sigmoid
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.5, pretrained_embeddings=None):
        """
        Initialize the LSTM sentiment classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pretrained_embeddings: Optional pretrained embeddings (numpy array)
        """
        super(LSTMSentimentClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            print("Loaded pretrained embeddings")
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, 1)
        
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Output tensor of shape (batch_size, 1) with probabilities
        """
        # Embedding: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)
        
        # LSTM: (batch_size, seq_length, embedding_dim) -> (batch_size, seq_length, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        # hidden shape: (num_layers, batch_size, hidden_dim)
        # We take the last layer's hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Apply dropout
        dropped = self.dropout(last_hidden)
        
        # Fully connected layer: (batch_size, hidden_dim) -> (batch_size, 1)
        output = self.fc(dropped)
        
        # Sigmoid activation
        output = self.sigmoid(output)
        
        return output.squeeze(1)  # (batch_size,)
    
    def predict(self, x):
        """
        Make predictions (for inference).
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted labels (0 or 1) and probabilities
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
            predictions = (probs >= 0.5).long()
        return predictions, probs


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    vocab_size = 25000
    batch_size = 32
    seq_length = 256
    
    model = LSTMSentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5
    )
    
    print(f"Model architecture:\n{model}")
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample outputs: {output[:5]}")