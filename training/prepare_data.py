"""
Clean the raw dataset and create the train / validation / test split.

    python -m training.prepare_data

Outputs
    data/processed/splits.csv                    one row per image with its split
    reports/evaluation/dataset_summary.json      counts per class and split + cleaning report
    reports/figures/class_distribution.png
"""

from __future__ import annotations

import argparse

from src import config as C
from src.data import (
    clean_images,
    find_dataset_root,
    list_images,
    save_splits,
    split_summary,
    stratified_split,
)
from src.evaluation import plot_class_distribution
from src.utils import get_logger, save_json

log = get_logger("prepare")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(C.RAW_DIR), help="folder that contains the extracted dataset")
    args = parser.parse_args()
    cfg = C.Config()

    root = find_dataset_root(args.raw_dir)
    log.info("Class folders found in: %s", root)

    df = list_images(root)
    log.info("Found %d image files in %d classes. Validating and de-duplicating (may take a minute) ...",
             len(df), df["label"].nunique())
    clean, report = clean_images(df)
    log.info("Cleaning report: %s", report)

    splits = stratified_split(clean, cfg.val_fraction, cfg.test_fraction, cfg.seed)
    save_splits(splits)

    summary = split_summary(splits)
    summary["cleaning_report"] = report
    summary["split_method"] = (
        f"stratified random split, seed={cfg.seed}: {1 - cfg.val_fraction - cfg.test_fraction:.0%} train / "
        f"{cfg.val_fraction:.0%} validation / {cfg.test_fraction:.0%} test"
    )
    save_json(summary, C.EVAL_DIR / "dataset_summary.json")
    plot_class_distribution(summary["per_class"], C.FIGURES_DIR / "class_distribution.png")

    log.info("Images per split: %s", summary["per_split"])
    for cls, counts in summary["per_class"].items():
        log.info("  %-10s %s", cls, counts)
    log.info("Saved %s", C.SPLITS_CSV)
    log.info("Next: python -m training.train --quick   (smoke test)  then  python -m training.train")


if __name__ == "__main__":
    main()
