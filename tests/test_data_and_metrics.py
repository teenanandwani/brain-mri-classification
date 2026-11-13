import numpy as np
import pytest
from PIL import Image

from src.data import build_manifest, load_image, split_manifest
from src.metrics import classification_metrics


def test_splits_keep_exact_duplicates_together(tmp_path):
    rng = np.random.default_rng(4)
    for name in ('no', 'yes'):
        folder = tmp_path / name
        folder.mkdir()
        for i in range(15):
            pixels = rng.integers(0, 255, size=(25, 27, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f'{i}.png')
            if i == 0:
                Image.fromarray(pixels).save(folder / 'duplicate.png')
    frame = build_manifest(tmp_path)
    assert len(frame) == 32
    train, val, test = split_manifest(frame)
    hashes = [set(x.image_hash) for x in (train, val, test)]
    assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
    assert sum(len(x) for x in (train, val, test)) == len(frame)
    arr = load_image(train.iloc[0]['path'], 64)
    assert arr.shape == (64, 64, 3) and arr.dtype == np.float32


def test_metrics_use_tumor_as_positive_class():
    m = classification_metrics([0, 0, 1, 1], [.1, .9, .8, .7], 0)
    assert m['confusion_matrix_no_yes'] == [[1, 1], [0, 2]]
    assert m['recall_tumor_sensitivity'] == 1.0
    assert m['specificity_no_tumor'] == .5
    assert m['accuracy'] == .75


def test_reject_conflicting_duplicate_labels(tmp_path):
    img = Image.fromarray(np.ones((20, 20, 3), dtype=np.uint8))
    for name in ('no', 'yes'):
        (tmp_path / name).mkdir()
        img.save(tmp_path / name / 'same.png')
    with pytest.raises(ValueError, match='conflicting'):
        build_manifest(tmp_path)
