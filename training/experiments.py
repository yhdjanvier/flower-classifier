"""
Compare a few hyper-parameter settings using VALIDATION results only.

    python -m training.experiments --quick   # smoke test
    python -m training.experiments           # real comparison (runs 3 trainings; be patient on CPU)

This is how "model parameters are adjusted based on evaluation results" is demonstrated:
  A  head only (no fine-tuning)          -> baseline
  B  fine-tune from layer 100, dropout 0.3 -> the default in src/config.py
  C  fine-tune from layer 60,  dropout 0.4 -> deeper fine-tuning with more regularisation

Results: reports/evaluation/experiments.csv.  Copy the winning values into src/config.py,
re-run `python -m training.train`, and only THEN evaluate once on the test set.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from src import config as C
from src.utils import get_logger, save_json
from training.train import run_training

log = get_logger("experiments")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    base = C.Config()
    if args.quick:
        base = replace(base, phase1_epochs=1, phase2_epochs=1, batch_size=16)

    plans = [
        ("A_head_only_do0.2", replace(base, dropout=0.2), False),
        ("B_finetune100_do0.3", replace(base, dropout=0.3, fine_tune_at=100), True),
        ("C_finetune60_do0.4", replace(base, dropout=0.4, fine_tune_at=60), True),
    ]
    rows = []
    for name, cfg, fine_tune in plans:
        log.info("=== Experiment %s ===", name)
        result = run_training(cfg, run_name=name, fine_tune=fine_tune, export_dir=None, reports_dir=None,
                              max_per_class=20 if args.quick else None)
        rows.append({
            "experiment": name, "dropout": cfg.dropout, "fine_tune": fine_tune,
            "fine_tune_at": cfg.fine_tune_at if fine_tune else None,
            "phase1_lr": cfg.phase1_lr, "phase2_lr": cfg.phase2_lr if fine_tune else None,
            "val_accuracy_phase1": result["val_phase1"]["accuracy"],
            "val_accuracy_final": result["val_final"]["accuracy"],
            "val_macro_f1_final": result["val_final"]["macro_f1"],
            "val_loss_final": result["val_final"]["loss"],
            "chosen_weights": result["chosen"], "minutes": result["minutes"],
        })

    table = pd.DataFrame(rows).sort_values("val_accuracy_final", ascending=False)
    out_dir = (C.REPORTS_DIR / "_quick_test") if args.quick else C.EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "experiments.csv", index=False)
    save_json(rows, out_dir / "experiments.json")
    print("\n", table.to_string(index=False))
    best = table.iloc[0]
    log.info("Best on validation: %s (accuracy %.4f). Put its settings in src/config.py and re-run training.",
             best["experiment"], best["val_accuracy_final"])


if __name__ == "__main__":
    main()
