"""Dataset discovery and deterministic, duplicate-aware train/val/test splits."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
CLASS_TO_ID = {'no': 0, 'yes': 1}


def locate_data_dir(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    for candidate in (root, root / 'brain_tumor_dataset'):
        if all((candidate / name).is_dir() for name in CLASS_TO_ID):
            return candidate
    raise FileNotFoundError(
        f'Expected subfolders "no" and "yes" under {root} '
        '(or its brain_tumor_dataset folder).'
    )


def decoded_pixel_hash(path: Path) -> str:
    """Find *exact* pixel duplicates even if their file encodings differ."""
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert('RGB')
        array = np.asarray(rgb)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode('ascii'))
    digest.update(array.tobytes())
    return digest.hexdigest()


def build_manifest(root: str | Path) -> pd.DataFrame:
    data_dir = locate_data_dir(root)
    records = []
    for class_name, label in CLASS_TO_ID.items():
        for path in sorted((data_dir / class_name).iterdir()):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                records.append({'path': str(path.resolve()),
                                'label': label, 'class_name': class_name,
                                'image_hash': decoded_pixel_hash(path)})
    if not records:
        raise ValueError(f'No images found in {data_dir}')
    df = pd.DataFrame(records)
    if df['label'].nunique() != 2:
        raise ValueError('Both "no" and "yes" classes are required.')
    conflicts = df.groupby('image_hash')['label'].nunique()
    if (conflicts > 1).any():
        raise ValueError('Identical image pixels found with conflicting no/yes labels. '
                         'Resolve the data before training.')
    return df


def split_manifest(df: pd.DataFrame, seed: int = 42):
    """Approx. 60/20/20 stratified splits without exact image duplication."""
    group_counts = df.drop_duplicates('image_hash').groupby('label').size()
    if len(group_counts) < 2 or group_counts.min() < 5:
        raise ValueError('At least five distinct images per class are required.')
    x = np.zeros(len(df))
    y = df['label'].to_numpy()
    groups = df['image_hash'].to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    development_idx, test_idx = next(outer.split(x, y, groups))
    development = df.iloc[development_idx].reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    dev_idx, val_idx = next(inner.split(
        np.zeros(len(development)), development['label'], development['image_hash']
    ))
    train, val, test = (
        development.iloc[dev_idx].reset_index(drop=True),
        development.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )
    for first, second in ((train, val), (train, test), (val, test)):
        assert set(first.image_hash).isdisjoint(set(second.image_hash)), 'Duplicate leakage'
    for name, split in (('train', train), ('val', val), ('test', test)):
        if split['label'].nunique() < 2:
            raise ValueError(f'{name} split has a single class; change seed or add data.')
    return train, val, test


def load_image(path: str | Path, size: int) -> np.ndarray:
    """EfficientNetB0 expects RGB float pixels in [0, 255]."""
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert('RGB')
        rgb = rgb.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.float32)
