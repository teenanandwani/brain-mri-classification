"""Train and evaluate an EfficientNetB0 tumor-presence classifier.

Run: python train.py --data-dir data/brain_tumor_dataset --epochs 15
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import build_manifest, load_image, split_manifest
from src.metrics import classification_metrics


def create_dataset(frame, size, batch_size, training, seed, tf):
    paths = frame['path'].astype(str).to_numpy()
    labels = frame['label'].astype('float32').to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _read(path, label):
        def _load(p):
            return load_image(p.decode('utf-8'), size)
        image = tf.numpy_function(_load, [path], tf.float32)
        image.set_shape((size, size, 3))
        return image, tf.reshape(label, (1,))

    ds = ds.map(_read, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(size, weights, tf):
    layers = tf.keras.layers
    # EfficientNet has its own input rescaling: pass images in 0-255, not 0-1.
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=(size, size, 3)
    )
    backbone.trainable = False
    inp = layers.Input(shape=(size, size, 3))
    x = tf.keras.Sequential([
        layers.RandomRotation(0.03), layers.RandomZoom(0.05),
        layers.RandomContrast(0.05),
    ], name='train_only_augmentation')(inp)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name='tumor_probability')(x)
    return tf.keras.Model(inp, out), backbone


def save_visuals(history, matrix, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history['loss'], label='train')
    axes[0].plot(history.history['val_loss'], label='validation')
    axes[0].set(title='Binary cross-entropy', xlabel='Epoch')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='train')
    axes[1].plot(history.history['val_accuracy'], label='validation')
    axes[1].set(title='Accuracy', xlabel='Epoch')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha='center', va='center')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['no', 'yes'],
           yticklabels=['no', 'yes'], xlabel='Predicted', ylabel='True',
           title='Held-out test confusion matrix')
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, help='Folder containing no/ and yes/')
    parser.add_argument('--output-dir', default='outputs/retrain')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weights', choices=['imagenet', 'none'], default='imagenet',
                        help='Use none only to smoke-test without a weight download.')
    args = parser.parse_args()
    if args.epochs < 1 or args.image_size < 32:
        parser.error('epochs must be >=1 and image-size must be >=32')

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit('TensorFlow is required for training. Run: pip install -r requirements.txt') from exc
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.data_dir)
    train, val, test = split_manifest(manifest, args.seed)
    for name, frame in (('train', train), ('val', val), ('test', test)):
        frame.to_csv(output / f'{name}_manifest.csv', index=False)
        print(name, len(frame), frame['class_name'].value_counts().to_dict())
    exact_duplicates = int(len(manifest) - manifest['image_hash'].nunique())
    print('Exact duplicate images in source (kept within one split):', exact_duplicates)
    print('Important: patient IDs are unavailable; related scans may still leak across splits.')
    train_ds = create_dataset(train, args.image_size, args.batch_size, True, args.seed, tf)
    val_ds = create_dataset(val, args.image_size, args.batch_size, False, args.seed, tf)
    test_ds = create_dataset(test, args.image_size, args.batch_size, False, args.seed, tf)

    weights = None if args.weights == 'none' else 'imagenet'
    model, _ = build_model(args.image_size, weights, tf)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.AUC(name='auc')])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2,
                                             factor=0.5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(str(output / 'best_model.keras'),
                                           save_best_only=True, monitor='val_loss'),
    ]
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs, callbacks=callbacks)
    # Test set is evaluated only after model selection has finished.
    best = tf.keras.models.load_model(output / 'best_model.keras', compile=False)
    probabilities = best.predict(test_ds, verbose=0).reshape(-1)
    majority = int(train['label'].mode().iloc[0])
    metrics = classification_metrics(test['label'].to_numpy(), probabilities, majority)
    metrics.update({
        'split': 'StratifiedGroupKFold by exact decoded pixel hash, NOT patient',
        'dataset_images': int(len(manifest)),
        'duplicate_image_count': exact_duplicates,
        'train_images': int(len(train)), 'validation_images': int(len(val)),
        'test_images': int(len(test)),
        'weights': args.weights,
        'seed': args.seed,
        'limitations': 'Small heterogeneous dataset with no patient identifiers. '
                       'Exact duplicates grouped, but related scans/near duplicates may remain.',
    })
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    predictions = test[['path', 'label']].copy()
    predictions['tumor_probability'] = probabilities
    predictions['predicted_label'] = (probabilities >= .5).astype(int)
    predictions.to_csv(output / 'test_predictions.csv', index=False)
    save_visuals(history, metrics['confusion_matrix_no_yes'], output)
    print(json.dumps({k: v for k, v in metrics.items() if k not in
                      ('classification_report', 'confusion_matrix_no_yes')}, indent=2))
    print(f'Artifacts saved in {output.resolve()}')


if __name__ == '__main__':
    main()
