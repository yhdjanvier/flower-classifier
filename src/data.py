"""
Dataset discovery, cleaning and splitting  (NO TensorFlow in this file).

Pipeline (run once by `python -m training.prepare_data`):

    raw folders  ->  list every image  ->  drop corrupted files
                 ->  drop duplicates   ->  stratified train/val/test split
                 ->  data/processed/splits.csv

DATA-LEAKAGE PROTECTION
  * The split is done on FILES, before any augmentation or pre-processing.
  * Byte-identical duplicate images are removed first (the public flower dataset
    contains some), so the same picture cannot land in both train and test.
  * `assert_no_leakage` re-checks this after the split and stops the program if violated.
  * Augmentation only exists inside the model and is switched off outside training,
    so validation/test images are never augmented.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src import config as C
from src.utils import get_logger

log = get_logger("data")


class DatasetNotFoundError(RuntimeError):
    """Raised when the raw dataset folder is missing or has an unexpected layout."""


# ----------------------------------------------------------------------------- discovery
def _image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in C.IMAGE_EXTENSIONS)


def find_dataset_root(raw_dir: Path = C.RAW_DIR) -> Path:
    """Find the folder whose sub-folders are the class folders (daisy/, rose/, ...).

    Kaggle zips are sometimes nested (flowers/flowers/daisy), so we search for the
    shallowest folder that contains at least two sub-folders holding images."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise DatasetNotFoundError(f"Dataset folder not found: {raw_dir}. Run `python -m training.download_dataset` first.")
    candidates = [raw_dir] + [p for p in raw_dir.rglob("*") if p.is_dir()]
    for folder in sorted(candidates, key=lambda p: (len(p.parts), p.as_posix())):
        class_dirs = [d for d in folder.iterdir() if d.is_dir() and _image_files(d)]
        if len(class_dirs) >= 2:
            return folder
    raise DatasetNotFoundError(
        f"No class folders with images were found under {raw_dir}. Expected e.g. {raw_dir / 'flowers' / 'daisy'}."
    )


def _store_path(path: Path) -> str:
    """Store paths relative to the project when possible (portable CSV), else absolute."""
    try:
        return path.resolve().relative_to(C.PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_path(stored: str) -> Path:
    p = Path(stored)
    return p if p.is_absolute() else C.PROJECT_ROOT / p


def list_images(root: Path) -> pd.DataFrame:
    """One row per image file: columns `path`, `label` (the class folder name)."""
    rows = []
    for class_dir in sorted(d for d in Path(root).iterdir() if d.is_dir()):
        for img in _image_files(class_dir):
            rows.append({"path": _store_path(img), "label": class_dir.name.strip().lower()})
    if not rows:
        raise DatasetNotFoundError(f"No images found under {root}.")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- cleaning
def is_valid_image(path: Path, min_side: int = C.MIN_DATASET_IMAGE_SIDE) -> bool:
    """True if Pillow can fully decode the file and it is not tiny."""
    try:
        with Image.open(path) as im:
            im.verify()                      # cheap structural check
        with Image.open(path) as im:
            im.load()                        # full decode catches truncated files
            width, height = im.size
            # CMYK/other exotic modes open in Pillow but TensorFlow's decoder rejects them mid-training
            if im.mode not in ("RGB", "L", "RGBA", "LA", "P"):
                return False
        return min(width, height) >= min_side
    except Exception:  # any decoding problem means "not usable"
        return False


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324  (used for duplicate detection, not security)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_images(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove corrupted files and duplicates. Returns (clean_df_with_md5, report)."""
    df = df.sort_values("path").reset_index(drop=True)
    report = {"found": int(len(df))}

    ok = [is_valid_image(resolve_path(p)) for p in df["path"]]
    report["corrupted_or_too_small_removed"] = int(len(df) - sum(ok))
    df = df[ok].copy()

    df["md5"] = [file_md5(resolve_path(p)) for p in df["path"]]

    # Same bytes filed under DIFFERENT classes -> label is ambiguous -> drop every copy.
    n_labels = df.groupby("md5")["label"].transform("nunique")
    conflict = n_labels > 1
    report["conflicting_label_duplicates_removed"] = int(conflict.sum())
    df = df[~conflict]

    # Same bytes, same class -> keep the first copy only.
    before = len(df)
    df = df.drop_duplicates(subset="md5", keep="first")
    report["exact_duplicates_removed"] = int(before - len(df))

    df = df.reset_index(drop=True)
    report["kept"] = int(len(df))
    return df, report


# ----------------------------------------------------------------------------- splitting
def stratified_split(df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int) -> pd.DataFrame:
    """Add a `split` column (train / val / test) keeping class proportions in every split."""
    df = df.sort_values("path").reset_index(drop=True)
    train_val, test = train_test_split(df, test_size=test_fraction, stratify=df["label"], random_state=seed)
    relative_val = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(train_val, test_size=relative_val, stratify=train_val["label"], random_state=seed)
    out = pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    ).sort_values("path").reset_index(drop=True)
    assert_no_leakage(out)
    return out


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Fail loudly if any file or any identical image content appears in two splits."""
    for column in ("path", "md5"):
        if column not in df.columns:
            continue
        per_split = df.groupby(column)["split"].nunique()
        leaked = per_split[per_split > 1]
        if len(leaked):
            raise AssertionError(f"Data leakage: {len(leaked)} items share `{column}` across splits.")


def split_summary(df: pd.DataFrame) -> dict:
    counts = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    for col in ("train", "val", "test"):
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["train", "val", "test"]]
    return {
        "classes": sorted(counts.index.tolist()),
        "total_images": int(len(df)),
        "per_split": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
        "per_class": {cls: {s: int(counts.loc[cls, s]) for s in counts.columns} for cls in counts.index},
    }


def save_splits(df: pd.DataFrame, path: Path = C.SPLITS_CSV) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_splits(path: Path = C.SPLITS_CSV) -> pd.DataFrame:
    """Load the split table and add an absolute-path column `abs_path`."""
    path = Path(path)
    if not path.exists():
        raise DatasetNotFoundError(f"{path} not found. Run `python -m training.prepare_data` first.")
    df = pd.read_csv(path)
    df["abs_path"] = [str(resolve_path(p)) for p in df["path"]]
    return df
