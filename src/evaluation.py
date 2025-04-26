import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, test_loader, device, model_path=None):
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    test_preds = []
    test_true_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            test_preds.extend(preds.cpu().tolist())
            test_true_labels.extend(labels.cpu().tolist())

    test_acc = accuracy_score(test_true_labels, test_preds)
    test_f1 = f1_score(test_true_labels, test_preds, average="macro")
    test_precision = precision_score(test_true_labels, test_preds, average="macro")
    test_recall = recall_score(test_true_labels, test_preds, average="macro")

    report = classification_report(test_true_labels, test_preds, output_dict=True)

    cm = confusion_matrix(test_true_labels, test_preds)

    results = {
        "accuracy": test_acc,
        "f1_score": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": test_preds,
        "true_labels": test_true_labels,
    }

    return results


def run_kfold_evaluation(
    model_class,
    train_df,
    test_df,
    prepare_fold_data,
    test_loader,
    device,
    num_folds=5,
    save_dir="models",
):
    from sklearn.model_selection import KFold
    import os

    os.makedirs(save_dir, exist_ok=True)

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df)):
        fold_num = fold + 1
        print(f"\n{'='*20} Fold {fold_num}/{num_folds} {'='*20}")

        fold_train_df = train_df.iloc[train_idx]
        fold_val_df = train_df.iloc[val_idx]

        print(
            f"Fold training set: {fold_train_df.shape}, Fold validation set: {fold_val_df.shape}"
        )

        fold_train_loader, fold_val_loader, fold_train_labels = prepare_fold_data(
            fold_train_df, fold_val_df
        )

        model = model_class.create_and_prepare_model(device=device)

        model_save_path = f"{save_dir}/fold_{fold_num}_model.pt"

        best_model_state, best_val_f1 = train_model(
            model,
            fold_train_loader,
            fold_val_loader,
            device,
            fold_train_labels,
            model_save_path=model_save_path,
        )

        model.load_state_dict(best_model_state)

        fold_test_results = evaluate_model(model, test_loader, device)

        print(f"\nFold {fold_num} Test Results:")
        print(f"Test Accuracy: {fold_test_results['accuracy']:.4f}")
        print(f"Test F1-score: {fold_test_results['f1_score']:.4f}")
        print(f"Test Precision: {fold_test_results['precision']:.4f}")
        print(f"Test Recall: {fold_test_results['recall']:.4f}")

        print("\nTest Classification Report:")
        print(
            classification_report(
                fold_test_results["true_labels"], fold_test_results["predictions"]
            )
        )

        print("\nConfusion Matrix:")
        print(fold_test_results["confusion_matrix"])

        fold_results.append(
            {
                "fold": fold_num,
                "test_acc": fold_test_results["accuracy"],
                "test_f1": fold_test_results["f1_score"],
                "test_precision": fold_test_results["precision"],
                "test_recall": fold_test_results["recall"],
                "test_preds": fold_test_results["predictions"],
                "test_true": fold_test_results["true_labels"],
            }
        )

    print("\n" + "=" * 50)
    print("K-Fold Cross Validation Results:")

    avg_test_acc = np.mean([res["test_acc"] for res in fold_results])
    avg_test_f1 = np.mean([res["test_f1"] for res in fold_results])
    avg_test_precision = np.mean([res["test_precision"] for res in fold_results])
    avg_test_recall = np.mean([res["test_recall"] for res in fold_results])

    print(f"Average Test Accuracy: {avg_test_acc:.4f}")
    print(f"Average Test F1-score: {avg_test_f1:.4f}")
    print(f"Average Test Precision: {avg_test_precision:.4f}")
    print(f"Average Test Recall: {avg_test_recall:.4f}")

    best_fold = np.argmax([res["test_f1"] for res in fold_results]) + 1
    print(f"Best performing model was from fold {best_fold}")

    final_model = model_class.create_and_prepare_model(device=device)
    final_model.load_state_dict(torch.load(f"{save_dir}/fold_{best_fold}_model.pt"))

    final_model_path = f"{save_dir}/final_model.pt"
    torch.save(final_model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    return {
        "fold_results": fold_results,
        "avg_test_acc": avg_test_acc,
        "avg_test_f1": avg_test_f1,
        "avg_test_precision": avg_test_precision,
        "avg_test_recall": avg_test_recall,
        "best_fold": best_fold,
        "final_model_path": final_model_path,
    }