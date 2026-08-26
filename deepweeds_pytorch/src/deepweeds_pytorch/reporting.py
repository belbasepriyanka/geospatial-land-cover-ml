from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(model, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred, filenames = [], [], []

    for images, labels, names in loader:
        logits = model(images.to(device))
        predictions = logits.argmax(dim=1).cpu().numpy().tolist()
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions)
        filenames.extend(list(names))

    return np.asarray(y_true), np.asarray(y_pred), filenames


def write_evaluation_outputs(
    y_true,
    y_pred,
    filenames,
    class_names: list[str],
    output_dir: str | Path,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(out / "classification_report.csv")

    pd.DataFrame(
        {"filename": filenames, "true_label": y_true, "predicted_label": y_pred}
    ).to_csv(out / "predictions.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="DeepWeeds confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    return metrics


def plot_training_history(history_csv: str | Path, output_path: str | Path) -> None:
    frame = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame["epoch"], frame["train_loss"], label="train loss")
    ax.plot(frame["epoch"], frame["val_loss"], label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
