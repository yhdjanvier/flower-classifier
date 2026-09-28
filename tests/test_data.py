"""Dataset cleaning + splitting (no TensorFlow needed)."""

import pandas as pd
import pytest
from PIL import Image

from src import data as D
from tests.conftest import make_image_bytes

CLASSES = ["daisy", "rose", "tulip"]


@pytest.fixture
def fake_dataset(tmp_path):
    root = tmp_path / "raw" / "flowers"
    for ci, cls in enumerate(CLASSES):
        (root / cls).mkdir(parents=True)
        for i in range(20):
            (root / cls / f"{cls}_{i}.jpg").write_bytes(make_image_bytes((64, 64), seed=ci * 100 + i))
    (root / "daisy" / "corrupt.jpg").write_bytes(b"this is not an image")
    (root / "daisy" / "copy_of_daisy_0.jpg").write_bytes((root / "daisy" / "daisy_0.jpg").read_bytes())  # same-class duplicate
    (root / "rose" / "daisy_1_mislabelled.jpg").write_bytes((root / "daisy" / "daisy_1.jpg").read_bytes())  # cross-class duplicate
    (root / "tulip" / "notes.txt").write_text("ignore me")
    Image.new("CMYK", (64, 64)).save(root / "tulip" / "cmyk.jpg")                                         # TF cannot decode CMYK
    return tmp_path / "raw"


def test_find_root_handles_nesting(fake_dataset):
    assert D.find_dataset_root(fake_dataset).name == "flowers"


def test_missing_dataset_message(tmp_path):
    with pytest.raises(D.DatasetNotFoundError):
        D.find_dataset_root(tmp_path / "nope")


def test_cleaning_removes_corrupt_and_duplicates(fake_dataset):
    df = D.list_images(D.find_dataset_root(fake_dataset))
    assert set(df["label"]) == set(CLASSES)
    assert not df["path"].str.endswith(".txt").any()
    clean, rep = D.clean_images(df)
    assert rep["corrupted_or_too_small_removed"] == 2          # garbage bytes + CMYK image
    assert rep["conflicting_label_duplicates_removed"] == 2      # both copies of the cross-class duplicate
    assert rep["exact_duplicates_removed"] == 1                  # copy_of_daisy_0
    assert clean["md5"].is_unique


def test_split_is_stratified_and_leak_free(fake_dataset):
    clean, _ = D.clean_images(D.list_images(D.find_dataset_root(fake_dataset)))
    out = D.stratified_split(clean, 0.15, 0.15, seed=42)
    assert set(out["split"]) == {"train", "val", "test"}
    assert out.groupby("md5")["split"].nunique().max() == 1
    assert out.groupby("path")["split"].nunique().max() == 1
    for cls in CLASSES:                                          # every class present in every split
        assert set(out[out["label"] == cls]["split"]) == {"train", "val", "test"}
    assert 0.6 < (out["split"] == "train").mean() < 0.8


def test_split_is_reproducible(fake_dataset):
    clean, _ = D.clean_images(D.list_images(D.find_dataset_root(fake_dataset)))
    a = D.stratified_split(clean, 0.15, 0.15, seed=1)
    b = D.stratified_split(clean, 0.15, 0.15, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_leakage_check_detects_problem():
    bad = pd.DataFrame({"path": ["a", "b"], "md5": ["x", "x"], "label": ["c", "c"], "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        D.assert_no_leakage(bad)
