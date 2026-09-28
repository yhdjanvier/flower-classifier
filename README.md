# 🌸 Flower Classifier - CNN Transfer Learning + Streamlit

A deep-learning web app that recognises the type of a flower from an uploaded photo.
Built for the *Python and Fundamentals of AI* practical assessment (Deep Learning / Computer Vision, TensorFlow/Keras).

> **Status of results:** this repository contains **no invented numbers**. Accuracy, precision, recall, F1 and the
> confusion matrix are produced when *you* run the training and evaluation scripts (sections 13-14, 19).

## 1-3. Problem, motivation, objectives
**Problem.** Given a photo, decide which of 5 flower types it shows (daisy, dandelion, rose, sunflower, tulip).
**Motivation.** Image recognition is a classic real-world AI task (plant apps, agriculture, education). Flower photos
are a compact, understandable example of a CNN learning visual features.
**Objectives.** (1) acquire and clean a public dataset; (2) build a transfer-learning CNN; (3) evaluate it properly on
unseen data; (4) improve it from evaluation feedback; (5) deploy it as a web app; (6) discuss limits and responsible use.

## 5-6. Dataset
| | |
|---|---|
| Name / source | *Flowers Recognition* (Kaggle, author `alxmamaev`) |
| URL | https://www.kaggle.com/datasets/alxmamaev/flowers-recognition |
| Classes | daisy, dandelion, rose, sunflower, tulip |
| Size | about 4,200-4,300 images (public sources quote 4,242 / 4,317 / 4,323 - the exact post-cleaning count is written to `reports/evaluation/dataset_summary.json`) |
| Characteristics | web photos, ~320x240 px, varying quality/lighting, some near-duplicates |
| Licence | check the Kaggle page; images are **not** stored in this repo |
| Split | stratified 70 % train / 15 % validation / 15 % test, seed 42, done on files *before* augmentation |
| Cleaning | unreadable/tiny files removed; byte-identical duplicates removed; identical images with different labels removed |

## 7-8. AI approach and why
**Approach:** CNN **transfer learning** with **MobileNetV2** (ImageNet-pretrained). The pretrained convolutional base is
a learned *feature extractor*; only a small classification head (and later the top backbone layers) is trained.
**Why:** ~4k images is too few to train a good CNN from scratch, but a network that already learned edges, textures and
shapes on 1.4 M ImageNet images transfers very well. MobileNetV2 was chosen over ResNet50/EfficientNetB0 because it is
small (~10 MB saved), fast on CPU, fits in GitHub, and loads quickly on Streamlit Cloud.

## 9. System architecture
Diagrams: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (inference, training, layers).

```
Upload -> validate -> resize 224x224 -> [augment: train only] -> rescale -1..1
       -> MobileNetV2 features -> GlobalAvgPool -> Dropout -> Dense(5) softmax -> prediction + confidence
```

## 10. Project structure
```
flower-classifier/
├── app.py                    Streamlit application
├── requirements.txt          runtime deps (Streamlit Cloud)
├── requirements-dev.txt      + training/test deps
├── pytest.ini  LICENSE  .gitignore  .env.example
├── .streamlit/config.toml
├── src/
│   ├── config.py             all paths + hyper-parameters
│   ├── data.py               discovery, cleaning, dedup, stratified split (no TensorFlow)
│   ├── preprocessing.py      tf.data pipeline + inference preprocessing (shared resize)
│   ├── model.py              MobileNetV2 model, augmentation, fine-tuning helpers
│   ├── predict.py            image validation + FlowerPredictor
│   ├── evaluation.py         metrics + plots (no TensorFlow)
│   └── utils.py
├── training/                 run as:  python -m training.<name>
│   ├── download_dataset.py   prepare_data.py   train.py   evaluate.py   experiments.py
├── models/                   flower_classifier.keras, class_names.json, config.json  (created by training; committed)
├── data/                     README only (dataset is re-created by scripts)
├── reports/{figures,evaluation}/   generated results (committed so the app can show them)
├── notebooks/exploration.ipynb
├── docs/                     ARCHITECTURE, RUBRIC_MAPPING, DEMO_SCRIPT, TROUBLESHOOTING
├── tests/                    pytest suite
└── PROJECT_REPORT.md
```

## 11. Environment setup (Windows + PyCharm)
**Python version.** Use **Python 3.12** if you can (the safest middle of TensorFlow's supported range; TensorFlow's install
page lists 3.10-3.13 for recent releases). If you only have 3.13 it should also work - just use the *same* minor version
locally and on Streamlit Cloud.

1. Copy/extract this project to `C:\Users\<YOUR_USERNAME>\PycharmProjects\flower-classifier`.
2. PyCharm -> **File -> Open** -> that folder. PyCharm offers to create a virtual environment: choose **Virtualenv**, base
   interpreter Python 3.12 (or 3.13), location `.venv` inside the project. (Or *Settings -> Project -> Python Interpreter -> Add Interpreter*.)
3. Open the terminal in PyCharm (bottom bar; it activates `.venv` automatically) and run:
```powershell
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
```
If PowerShell refuses to activate `.venv` outside PyCharm: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
CPU is enough; no NVIDIA GPU is required (native-Windows TensorFlow is CPU-only anyway).

**Pin your versions (important, do once after it works):**
```powershell
python -m pip freeze | findstr /i "tensorflow keras streamlit pillow numpy pandas"
```
Copy the exact `tensorflow-cpu==x.y.z` line into `requirements.txt` so Streamlit Cloud installs the same TensorFlow/Keras
that saved your model.

## 12. Dataset setup
```powershell
python -m training.download_dataset
python -m training.prepare_data
```
Automatic download uses `kagglehub`. Credentials (only if Kaggle asks): Kaggle -> *Settings -> API* -> create a token, put the
downloaded `kaggle.json` in `C:\Users\<YOUR_USERNAME>\.kaggle\kaggle.json` (never inside the project), or set
`$env:KAGGLE_USERNAME` / `$env:KAGGLE_KEY`. **Manual alternative (always works):** download the zip from the Kaggle page and extract to
`data\raw\flowers\<class>\*.jpg`, then run `prepare_data`.

## 13. Training
```powershell
python -m training.train --quick     # smoke test (small subset, 1 epoch per phase) - run this FIRST
python -m training.train             # real run; needs internet once to fetch ImageNet weights
```
* Phase 1: backbone frozen, train head (Adam, lr 1e-3, up to 15 epochs, early stopping).
* Phase 2: unfreeze backbone layers >= 100 (BatchNorm kept frozen), lr 1e-5, up to 10 epochs.
* The better phase (by **validation** accuracy) is kept. Output: `models/flower_classifier.keras`, `class_names.json`, `config.json`,
  `reports/evaluation/{training_history,model_selection}.json`, `reports/figures/training_curves.png`.
* Duration on CPU depends on your machine (not measured here) - the `--quick` run tells you roughly how fast it is.

**Improving the model from evaluation results** (marks: parameters adjusted):
```powershell
python -m training.experiments       # compares A head-only / B fine-tune@100 / C fine-tune@60 + more dropout on VALIDATION data
```
Read `reports/evaluation/experiments.csv`, copy the winning values into `src/config.py`, re-run `train`. Record your
before/after numbers in `PROJECT_REPORT.md` section 15. The test set is **not** used for tuning.

## 14. Evaluation
```powershell
python -m training.evaluate          # ONE evaluation on the unseen test split
```
Creates `test_metrics.json`, `test_classification_report.txt`, `test_misclassified.csv`, and confusion-matrix figures.
Smoke-test variant after `train --quick`: `python -m training.evaluate --model-dir models/_quick_test --reports-dir reports/_quick_test --split val`.

## 15. Run Streamlit
```powershell
streamlit run app.py
```
Opens http://localhost:8501. Sidebar: confidence threshold and number of predictions. Tabs: Classify, Model performance, About & responsible use.

## 16. Testing
```powershell
python -m pytest -q
```
Covers preprocessing, invalid images, class labels, prediction format, confidence, model loading, cleaning/splitting/leakage, metrics and an app smoke test.
Manual checklist: [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md#manual-test-checklist).

## 17. GitHub setup
1. Create an **empty** repository on github.com (no README/licence/.gitignore), e.g. `flower-classifier`.
2. In the PyCharm terminal:
```powershell
git init
git add .
git status
```
Check that `data/raw`, `.venv`, `.idea`, `kaggle.json` are **not** listed, and that `models/flower_classifier.keras` **is**.
```powershell
dir models
git commit -m "Flower classifier: CNN transfer learning + Streamlit"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/flower-classifier.git
git push -u origin main
```
First-time Git may ask for `git config --global user.name "..."` and `user.email "..."`. GitHub asks you to sign in via the browser.

**Model storage.** The exported model is small (MobileNetV2 + tiny head; expected around 10 MB - check with `dir models`), well under
GitHub's 100 MB file limit, so it is committed directly and Streamlit loads it from the repo: no retraining on startup, no external host.
*Fallback if your file is ever too large:* upload it as a GitHub Release asset (or any public HTTPS host) and add
`MODEL_URL = "https://..."` in Streamlit *Settings -> Secrets*; `app.py` downloads it once at startup.

## 18. Streamlit deployment
1. Go to https://share.streamlit.io and sign in with GitHub (authorise access to your repo).
2. **Create app** -> *Deploy a public app from GitHub*.
3. Repository: `<you>/flower-classifier` - Branch: `main` - Main file path: `app.py`.
4. **Advanced settings** -> choose the same Python version you used locally.
5. **Deploy**. First build takes several minutes (TensorFlow).
6. Open the public URL and run the manual checklist. Logs: *Manage app*.
No secrets are required (only the optional `MODEL_URL`). Pushing to GitHub redeploys automatically.

## 19. Results
> Not generated yet. After running the scripts, paste your real numbers here.

| Metric (test set) | Value |
|---|---|
| Accuracy | *run `python -m training.evaluate`* |
| Macro precision / recall / F1 | *…* |
| Most confused classes | *see `test_metrics.json` -> `top_confusions`* |

| Model improvement (validation set) | Accuracy | Macro-F1 |
|---|---|---|
| Phase 1 (head only) | *from `model_selection.json`* | |
| Phase 2 (fine-tuned) | | |

## 20-21. Responsible AI and limitations
Educational demo only - **not** a botanical identification system and never a basis for deciding whether a plant is safe to eat or touch.
Only 5 classes (other flowers/non-flowers are forced into one of them); web-sourced photos may be biased (region, variety, lighting, framing);
poor or unusual images reduce accuracy; confidence is not correctness; the model can be wrong and the UI says so when confidence is low.
Reduction: threshold + warning, quality warnings, honest wording, evaluating per class, no images stored, plans for more classes/data (below).

## 22. Future improvements
More flower classes (e.g. Oxford 102), an "unknown/not a flower" class or out-of-distribution detection, probability calibration, Grad-CAM heat-maps,
TensorFlow Lite for lighter deployment, mobile camera capture.

## 23. Troubleshooting
See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

## 24. Licence / attribution
Code: MIT ([LICENSE](LICENSE)). Dataset: Kaggle *Flowers Recognition* - see its page for terms. Model: MobileNetV2 (Sandler et al., CVPR 2018), ImageNet weights via Keras Applications.
