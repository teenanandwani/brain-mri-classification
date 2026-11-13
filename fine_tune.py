"""Fine-tune the already-trained EfficientNetB0 using SAME train/validation split.

Does not read the held-out test manifest or compute new test metrics.
Run after copying your local baseline outputs to experiments/baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train import create_dataset
from src.metrics import classification_metrics


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {'path', 'label', 'image_hash'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Invalid manifest {path}: expected {sorted(required)}')
    missing = [x for x in frame.path if not Path(x).exists()]
    if missing:
        raise FileNotFoundError(f'Image path not found: {missing[0]} (keep the Kaggle cache or re-download data)')
    return frame


def find_backbone(model, tf):
    candidates = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)
                  and 'efficientnet' in layer.name.lower()]
    if len(candidates) != 1:
        raise ValueError('Expected one EfficientNet backbone, found: '
                         + ', '.join(x.name for x in candidates))
    return candidates[0]


def unfreeze_last_layers(backbone, number, tf):
    if number < 1 or number > len(backbone.layers):
        raise ValueError(f'--unfreeze-last must be between 1 and {len(backbone.layers)}')
    backbone.trainable = True
    for layer in backbone.layers[:-number]:
        layer.trainable = False
    for layer in backbone.layers[-number:]:
        # Batch normalization uses training statistics; keep these frozen.
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    return [layer.name for layer in backbone.layers if layer.trainable]


def eval_validation(model, ds, val_labels, majority):
    probs = model.predict(ds, verbose=0).reshape(-1)
    return classification_metrics(val_labels, probs, majority), probs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, default=Path('experiments/baseline'))
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/fine_tuned'))
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--unfreeze-last', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error('--epochs must be >=1 and --learning-rate must be >0')
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit('Install project requirements before fine-tuning.') from exc
    tf.keras.utils.set_random_seed(args.seed)
    b = args.baseline_dir
    checkpoint = b / 'best_model.keras'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Missing {checkpoint}. Copy your ORIGINAL outputs folder first.')
    train = load_manifest(b / 'train_manifest.csv')
    val = load_manifest(b / 'val_manifest.csv')
    if set(train.image_hash) & set(val.image_hash):
        raise ValueError('Exact image leakage between train and validation manifests.')
    model = tf.keras.models.load_model(checkpoint, compile=False)
    train_ds = create_dataset(train, args.image_size, args.batch_size, True, args.seed, tf)
    val_ds = create_dataset(val, args.image_size, args.batch_size, False, args.seed, tf)
    majority = int(train.label.mode().iloc[0])
    before, before_probs = eval_validation(model, val_ds, val.label.to_numpy(), majority)

    backbone = find_backbone(model, tf)
    unfrozen_names = unfreeze_last_layers(backbone, args.unfreeze_last, tf)
    print(f'Fine-tuning {len(unfrozen_names)} non-BatchNorm backbone layers (plus the head).')
    print('Note: this does not inspect or evaluate the test set.')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')])
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(output / 'best_model.keras'),
                                               monitor='val_loss', save_best_only=True),
        ],
    )
    selected = tf.keras.models.load_model(output / 'best_model.keras', compile=False)
    after, after_probs = eval_validation(selected, val_ds, val.label.to_numpy(), majority)
    comparison = {
        'warning': 'Validation metrics only. The previously viewed small test set is NOT an unbiased final comparison.',
        'baseline_frozen': before,
        'fine_tuned': after,
        'config': {'unfreeze_last': args.unfreeze_last, 'learning_rate': args.learning_rate,
                   'epochs_requested': args.epochs, 'epochs_run': len(history.history['loss']),
                   'train_images': len(train), 'validation_images': len(val)},
    }
    (output / 'validation_comparison.json').write_text(json.dumps(comparison, indent=2))
    summary = pd.DataFrame([
        {'model': 'Frozen baseline', 'balanced_accuracy': before['balanced_accuracy'],
         'sensitivity': before['recall_tumor_sensitivity'], 'specificity': before['specificity_no_tumor'],
         'roc_auc': before['roc_auc']},
        {'model': 'Fine-tuned', 'balanced_accuracy': after['balanced_accuracy'],
         'sensitivity': after['recall_tumor_sensitivity'], 'specificity': after['specificity_no_tumor'],
         'roc_auc': after['roc_auc']},
    ])
    summary.to_csv(output / 'validation_comparison.csv', index=False)
    pred = val[['path', 'label']].copy()
    pred['baseline_score'] = before_probs
    pred['fine_tuned_score'] = after_probs
    pred.to_csv(output / 'validation_predictions.csv', index=False)  # local only
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history['loss'], label='training')
    axes[0].plot(history.history['val_loss'], label='validation')
    axes[0].set(title='Fine-tuning cross-entropy', xlabel='Epoch')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='training')
    axes[1].plot(history.history['val_accuracy'], label='validation')
    axes[1].set(title='Fine-tuning accuracy', xlabel='Epoch')
    axes[1].legend()
    fig.tight_layout(); fig.savefig(output / 'fine_tuning_curves.png', dpi=150); plt.close(fig)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('Results:', output.resolve())


if __name__ == '__main__':
    main()
