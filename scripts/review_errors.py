"""Visualize incorrect predictions from a LOCAL CSV; images are never uploaded.

Usage: python scripts/review_errors.py --csv experiments/baseline/test_predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps


def select_errors(frame: pd.DataFrame) -> pd.DataFrame:
    required = {'path', 'label', 'tumor_probability', 'predicted_label'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    return frame.loc[frame['label'] != frame['predicted_label']].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path('experiments/baseline/test_predictions.csv'))
    parser.add_argument('--output', type=Path, default=Path('experiments/baseline/error_review.png'))
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    mistakes = select_errors(frame)
    print(f'Incorrect predictions: {len(mistakes)} out of {len(frame)}')
    print('These are TEST mistakes for qualitative analysis, NOT for tuning decisions.')
    if mistakes.empty:
        print('No errors to plot.'); return
    rows, cols = (len(mistakes) + 2) // 3, min(3, len(mistakes))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.0 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, (_, record) in zip(axes.flat, mistakes.iterrows()):
        path = Path(str(record.path))
        if not path.exists():
            print(f'Not found: {path}'); ax.set_title(f'File missing: {path.name}'); continue
        with Image.open(path) as im:
            ax.imshow(ImageOps.exif_transpose(im).convert('RGB'))
        actual = 'tumor' if record.label == 1 else 'no tumor'
        predicted = 'tumor' if record.predicted_label == 1 else 'no tumor'
        ax.set_title(f'{path.name[:24]}\ntrue={actual}, pred={predicted}\nscore={record.tumor_probability:.2f}', fontsize=9)
    fig.suptitle('Misclassified images — local inspection only', fontsize=14)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print('Saved:', args.output)
    print('Check: unusual crop? text or labels? orientation? scan type? artifacts?')


if __name__ == '__main__':
    main()
