
import torch
import argparse
from model import LSTMSentimentClassifier
from utils import load_model


def preprocess_text(text, vocab, max_length=256):
    # Encode text
    encoded = vocab.encode(text)
    
    # Pad or truncate
    if len(encoded) < max_length:
        encoded = encoded + [0] * (max_length - len(encoded))
    else:
        encoded = encoded[:max_length]
    
    # Convert to tensor and add batch dimension
    tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    
    return tensor


def predict_sentiment(text, model, vocab, device):
    # Preprocess text
    input_tensor = preprocess_text(text, vocab).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()
        prediction = "Positive" if probability >= 0.5 else "Negative"
    
    return prediction, probability


def interactive_mode(model, vocab, device):
    print("\n" + "="*60)
    print("IMDB Sentiment Analysis - Interactive Mode")
    print("="*60)
    print("\nEnter movie reviews to analyze their sentiment.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        # Get user input
        print("-" * 60)
        text = input("\nEnter a movie review: ").strip()
        
        # Check for exit command
        if text.lower() in ['quit', 'exit', 'q']:
            print("\nExiting interactive mode. Goodbye!")
            break
        
        # Skip empty input
        if not text:
            print("Please enter a valid review.")
            continue
        
        # Make prediction
        prediction, probability = predict_sentiment(text, model, vocab, device)
        
        # Display results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Review: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"\nPredicted Sentiment: {prediction}")
        print(f"Confidence: {probability:.4f}")
        
        # Add interpretation
        if probability >= 0.8 or probability <= 0.2:
            confidence_level = "Very confident"
        elif probability >= 0.65 or probability <= 0.35:
            confidence_level = "Confident"
        else:
            confidence_level = "Uncertain"
        
        print(f"Interpretation: {confidence_level} {prediction.lower()} sentiment")
        print("="*60)


def batch_mode(texts, model, vocab, device):
    print("\n" + "="*60)
    print("IMDB Sentiment Analysis - Batch Mode")
    print("="*60)
    print(f"\nAnalyzing {len(texts)} reviews...\n")
    
    for i, text in enumerate(texts, 1):
        prediction, probability = predict_sentiment(text, model, vocab, device)
        
        print(f"Review {i}:")
        print(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  Sentiment: {prediction}")
        print(f"  Confidence: {probability:.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Test sentiment classifier on custom input')
    parser.add_argument('--text', type=str, help='Single text to analyze')
    parser.add_argument('--file', type=str, help='File containing reviews (one per line)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--model_dir', type=str, default='saved_models', 
                       help='Directory containing saved model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model and vocabulary
    print("\nLoading model and vocabulary...")
    model, vocab = load_model(LSTMSentimentClassifier, save_dir=args.model_dir, device=device)
    print("Model loaded successfully!\n")
    
    # Determine mode
    if args.text:
        # Single text mode
        print("="*60)
        print("Single Text Analysis")
        print("="*60)
        prediction, probability = predict_sentiment(args.text, model, vocab, device)
        print(f"\nReview: {args.text}")
        print(f"\nPredicted Sentiment: {prediction}")
        print(f"Confidence: {probability:.4f}")
        print("="*60)
        
    elif args.file:
        # Batch mode from file
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            batch_mode(texts, model, vocab, device)
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.")
        except Exception as e:
            print(f"Error reading file: {e}")
            
    else:
        # Interactive mode (default)
        interactive_mode(model, vocab, device)


if __name__ == "__main__":
    main()
