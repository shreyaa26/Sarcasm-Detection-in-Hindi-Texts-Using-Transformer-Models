# Sarcasm Detection in Hindi Texts Using Transformer Models

The project implements and evaluates four different transformer models - IndicBERT, mBERT, XLM-RoBERTa, and MuRIL - on a dataset of pure Hindi tweets, for the classification task of sarcasm detection.

## Setup

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/shreyaa26/Sarcasm-Detection-in-Hindi-Texts-Using-Transformer-Models.git
cd Sarcasm-Detection-in-Hindi-Texts-Using-Transformer-Models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training and Evaluation

The main script supports training and evaluating models with various configurations:

```bash
python main.py --data_path path/to/your/data.csv --model_type indicbert
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to the dataset (CSV file) | Required |
| `--model_type` | Type of model to use (indicbert, muril, xlm, mbert) | indicbert |
| `--num_folds` | Number of folds for cross-validation | 5 |
| `--max_epochs` | Maximum number of epochs for training | 5 |
| `--batch_size` | Batch size for training | 16 |
| `--save_dir` | Directory to save models | models |
| `--use_augmentation` | Flag to use text augmentation | False |
| `--augment_ratio` | Ratio of data to augment | 0.3 |

### Data Format

The input CSV file should contain at least two columns:
- `text`: The input text to classify
- `label`: Binary label (0 for non-sarcastic, 1 for sarcastic)

## Models

The project implements four transformer models:

1. **IndicBERT**: A BERT model pre-trained on Indic languages by AI4Bharat
2. **MuRIL**: A transformer-based language model from Google specifically pre-trained on 17 Indian languages
3. **XLM-RoBERTa**: A cross-lingual RoBERTa model
4. **mBERT**: Multilingual BERT

All models follow a similar architecture:
- Pre-trained transformer as a base
- Custom classification head
- Strategic layer freezing to prevent overfitting

## Implementation Details

### Key Features

- **K-fold Cross-validation**: Ensures robust model evaluation
- **Cross Entropy Loss**: Better handling of class imbalance
- **Text Augmentation**: Improves model generalization
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Optimizes training process

### Training Process

The training pipeline includes:
1. Data preprocessing and tokenization
2. K-fold cross-validation setup
3. Model training with early stopping
4. Evaluation on validation set for each fold
5. Final evaluation on test set
6. Selection of best model across folds

## Results

Models are evaluated using:
- Accuracy
- F1-score (macro)
- Precision
- Recall
- Confusion matrix

Results are displayed after training and saved in the specified model directory.
