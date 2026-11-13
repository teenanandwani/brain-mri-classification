"""Test-set metrics; no TensorFlow dependency, so these are easy to unit-test."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)


def classification_metrics(y_true, tumor_probability, majority_baseline_label):
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(tumor_probability, dtype=float)
    if len(y) == 0 or y.shape != probabilities.shape:
        raise ValueError('Labels and predictions must be nonempty, matching vectors.')
    predictions = (probabilities >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        'sample_count': int(len(y)),
        'accuracy': float(accuracy_score(y, predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(y, predictions)),
        'precision_tumor': float(precision_score(y, predictions, zero_division=0)),
        'recall_tumor_sensitivity': float(recall_score(y, predictions, zero_division=0)),
        'specificity_no_tumor': float(tn / (tn + fp)) if tn + fp else None,
        'f1_tumor': float(f1_score(y, predictions, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, probabilities)) if len(set(y)) == 2 else None,
        'average_precision': float(average_precision_score(y, probabilities)) if len(set(y)) == 2 else None,
        'majority_class_baseline_accuracy': float(np.mean(y == majority_baseline_label)),
        'confusion_matrix_no_yes': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'classification_report': classification_report(
            y, predictions, labels=[0, 1], target_names=['no', 'yes'],
            zero_division=0, output_dict=True,
        ),
    }
