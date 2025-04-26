import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df


def create_train_test_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Create DataFrames from the splits
    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    return train_df, test_df


def prepare_kfold_data(train_df, test_df, tokenizer, max_length=128, batch_size=16):
    from torch.utils.data import TensorDataset, DataLoader
    import torch

    # Extract test data
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    # Tokenize test data
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    test_dataset = TensorDataset(
        test_encodings["input_ids"],
        test_encodings["attention_mask"],
        torch.tensor(test_labels),
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    def prepare_fold_data(fold_train_df, fold_val_df):
        # Extract texts and labels
        fold_train_texts = fold_train_df["text"].tolist()
        fold_train_labels = fold_train_df["label"].tolist()
        fold_val_texts = fold_val_df["text"].tolist()
        fold_val_labels = fold_val_df["label"].tolist()

        # Tokenize data
        fold_train_encodings = tokenizer(
            fold_train_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        fold_train_dataset = TensorDataset(
            fold_train_encodings["input_ids"],
            fold_train_encodings["attention_mask"],
            torch.tensor(fold_train_labels),
        )

        fold_train_loader = DataLoader(
            fold_train_dataset, batch_size=batch_size, shuffle=True
        )

        fold_val_encodings = tokenizer(
            fold_val_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        fold_val_dataset = TensorDataset(
            fold_val_encodings["input_ids"],
            fold_val_encodings["attention_mask"],
            torch.tensor(fold_val_labels),
        )

        fold_val_loader = DataLoader(fold_val_dataset, batch_size=batch_size)

        return fold_train_loader, fold_val_loader, fold_train_labels

    return test_loader, prepare_fold_d
