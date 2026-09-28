# Marking-rubric traceability (30 marks)

| # | Assessment criterion | Marks | Evidence in project | File / location |
|---|---|---|---|---|
| **APPLY DATA PREPROCESSING (30 % = 9 marks)** |||||
| 1 | Environment properly configured | 1.5 | PyCharm venv, pinned/ranged deps, Python version stated, TensorFlow import check | `requirements.txt`, `requirements-dev.txt`, README §11 |
| 2 | Functionalities specified | 1.5 | Numbered functional list: upload, validate, preprocess, predict, top-N, uncertainty, error handling, performance tab | README §1-3, §15; `PROJECT_REPORT.md` §2-4, §17 |
| 3 | Data acquired from the given source | 3 | Kaggle dataset, scripted + manual acquisition, documented source/URL/classes | `training/download_dataset.py`, `data/README.md`, README §5-6 |
| 4 | Data pre-processed | 3 | Validation, de-duplication, stratified split, resize, tf.data batching/prefetch, seeds | `src/data.py`, `training/prepare_data.py`, `src/preprocessing.py` |
| **APPLY DEEP LEARNING ALGORITHMS (50 % = 15 marks)** |||||
| 5 | Data features engineered | 3 | Pixels -> 224x224 float tensors -> scaled to [-1,1]; augmentation (flip, rotation, zoom, shift, brightness, contrast) | `src/preprocessing.py`, `src/model.py::build_augmentation`, `PROJECT_REPORT.md` §7-8 |
| 6 | Model features engineered | 3 | Pretrained convolutional base as learned feature extractor -> global average pooling -> dropout -> dense head | `src/model.py::build_model` |
| 7 | AI approach/model selected and justified | 1.5 | CNN transfer learning with MobileNetV2; comparison vs ResNet50/EfficientNetB0 | README §7-8, `PROJECT_REPORT.md` §9-10 |
| 8 | Important parameters selected and explained | 1.5 | Every hyper-parameter in one commented dataclass; explanation table | `src/config.py`, `PROJECT_REPORT.md` §11 |
| 9 | Application correctly implemented and tested | 6 | Two-phase training, evaluation, Streamlit app, pytest suite, manual checklist | `training/train.py`, `app.py`, `tests/`, `docs/DEMO_SCRIPT.md` |
| **APPLY MODEL EVALUATION TECHNIQUES (20 % = 6 marks)** |||||
| 10 | Appropriate metrics selected | 1 | Accuracy, precision, recall, F1 (macro/weighted/per-class), confusion matrix, curves | `src/evaluation.py`, `PROJECT_REPORT.md` §13 |
| 11 | Evaluation implemented, results interpreted | 2 | Test-set evaluation, saved JSON/report/figures, top confusions, interpretation guidance | `training/evaluate.py`, `reports/`, `PROJECT_REPORT.md` §14 |
| 12 | Parameters adjusted from evaluation | 1 | Phase 1 vs Phase 2 comparison saved automatically; 3-way experiment table on validation | `training/experiments.py`, `reports/evaluation/model_selection.json`, `experiments.csv` |
| 13 | Model/config saved for reproducibility | 1 | `.keras` model, `class_names.json`, `config.json`, history, split table logic + seed | `models/`, `reports/evaluation/training_history.json` |
| 14 | Deployed/demonstrated via web app | 1 | Streamlit app locally + Streamlit Community Cloud | `app.py`, README §15, §18 |
| — | Responsible use (task requirement 8) | — | Limitations + mitigations in UI, README, report | `app.py` "About" tab, README §20, `PROJECT_REPORT.md` §19-20 |

> Numbers in your defence (accuracy etc.) must come from *your* runs, not from this document.
