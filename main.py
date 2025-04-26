import torch
import argparse
import pandas as pd
from transformers import AutoTokenizer
    
from models.indicbert_model import SarcasmDetector
from models.muril_model import MuRILSarcasmDetector
from models.xlm_model import XLMModel
from models.mbert_model import MBERTModel
from src.preprocessing import create_train_test_split, prepare_kfold_data
from src.augmentation import TextAugmenter
from src.training import train_model
from src.evaluation import run_kfold_evaluation

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate a sarcasm detection model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model_type', type=str, default='indicbert', choices=['indicbert', 'bert', 'xlm', 'mbert'],
                        help='Type of model to use')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--max_epochs', type=int, default=5, help='Maximum number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--use_augmentation', action='store_true', help='Use text augmentation')
    parser.add_argument('--augment_ratio', type=float, default=0.3, help='Ratio of data to augment')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    df = pd.read_csv(args.data_path)
    X = df['text'].tolist()
    y = df['label'].tolist()
    
    # Create train-test split
    train_df, test_df = create_train_test_split(X, y, test_size=0.2)
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    # Apply augmentation if requested
    if args.use_augmentation:
        augmenter = TextAugmenter()
        aug_texts, aug_labels = augmenter.apply_augmentations(
            train_df['text'].tolist(), 
            train_df['label'].tolist(),
            augment_ratio=args.augment_ratio
        )
        train_df = pd.DataFrame({'text': aug_texts, 'label': aug_labels})
        print(f"Augmented training set shape: {train_df.shape}")
    
    # Set up the tokenizer based on model type
    model_map = {
        'indicbert': 'ai4bharat/indic-bert',
        'muril': 'google/muril-base-cased',
        'xlm': 'xlm-roberta-base',
        'mbert': 'bert-base-multilingual-cased'
    }
    
    model_name = model_map[args.model_type]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare data for k-fold
    test_loader, prepare_fold_data = prepare_kfold_data(
        train_df, test_df, tokenizer, 
        max_length=128, batch_size=args.batch_size
    )
    
    # Run k-fold cross-validation and evaluation
    results = run_kfold_evaluation(
        SarcasmDetector, train_df, test_df, prepare_fold_data, 
        test_loader, device, num_folds=args.num_folds, save_dir=args.save_dir
    )
    
    print("\nTraining and evaluation completed!")
    print(f"Best model saved to: {results['final_model_path']}")

if __name__ == "__main__":
    main()