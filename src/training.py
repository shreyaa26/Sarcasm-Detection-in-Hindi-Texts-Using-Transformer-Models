import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import get_linear_schedule_with_warmup


def train_model(
    model,
    fold_train_loader,
    fold_val_loader,
    device,
    fold_train_labels,
    max_epochs=5,
    patience=3,
    model_save_path=None,
):
    # Calculate class weights
    unique_labels = np.unique(fold_train_labels)
    class_weights = compute_class_weight(
        "balanced", classes=unique_labels, y=fold_train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Define optimizer with different learning rates
    bert_params = [
        p for n, p in model.named_parameters() if "bert" in n and p.requires_grad
    ]
    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n]

    # Adding weight decay for L2 regularization
    optimizer = torch.optim.AdamW(
        [
            {
                "params": bert_params,
                "lr": 1e-5,
                "weight_decay": 0.01,
            },  # Lower learning rate for BERT with L2 reg
            {
                "params": classifier_params,
                "lr": 2e-4,
                "weight_decay": 0.01,
            },  # Higher learning rate for classifier with L2 reg
        ]
    )

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Add learning rate scheduler
    total_steps = len(fold_train_loader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # Early stopping parameters
    counter = 0
    best_val_f1 = 0
    best_model_state = None

    # Training loop with early stopping
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_true_labels = []

        progress_bar = tqdm(
            fold_train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Training]"
        )
        for batch in progress_bar:
            # Move batch to device
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Update tracking variables
            train_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            train_preds.extend(preds.cpu().tolist())
            train_true_labels.extend(labels.cpu().tolist())

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

        # Calculate training metrics
        train_acc = accuracy_score(train_true_labels, train_preds)
        train_f1 = f1_score(train_true_labels, train_preds, average="macro")
        avg_train_loss = train_loss / len(fold_train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true_labels = []

        with torch.no_grad():
            for batch in tqdm(
                fold_val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Validation]"
            ):
                # Move batch to device
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels = batch

                # Forward pass
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)

                # Update tracking variables
                val_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_true_labels.extend(labels.cpu().tolist())

        # Calculate validation metrics
        val_acc = accuracy_score(val_true_labels, val_preds)
        val_f1 = f1_score(val_true_labels, val_preds, average="macro")
        val_precision = precision_score(val_true_labels, val_preds, average="macro")
        val_recall = recall_score(val_true_labels, val_preds, average="macro")
        avg_val_loss = val_loss / len(fold_val_loader)

        # Print epoch statistics
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        print(
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}"
        )
        print(
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
        )
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")

        # Save best model based on validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            counter = 0  # Reset early stopping counter
        else:
            counter += 1  # Increment counter if validation didn't improve

        # Early stopping check
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Save the best model if path is provided
    if model_save_path and best_model_state:
        torch.save(best_model_state, model_save_path)
        print(f"Best model saved to {model_save_path}")

    return best_model_state, best_val_f1