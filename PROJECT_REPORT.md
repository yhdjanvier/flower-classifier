# Flower Image Classification with CNN Transfer Learning - Project Report

*Module: Python and Fundamentals of AI (ITLPA701) - Practical assessment.*
Sections are tagged with the rubric criterion they support, e.g. **[Rubric 4]** (see `docs/RUBRIC_MAPPING.md`).
Anything written as **‹fill›** must be completed with numbers from your own runs. No results are invented here.

---
## 1. Introduction
Computer vision lets computers interpret photographs. This project builds and deploys a deep-learning application that identifies a flower type from an image.

## 2. Problem statement **[Rubric 2]**
Given one photograph, predict which of five flower categories (daisy, dandelion, rose, sunflower, tulip) it shows, report how confident the model is, and warn the user when it is not sure.

## 3. Problem domain
Image recognition (computer vision) - one of the domains listed in the task requirements. Real-world relevance: plant-identification apps, agriculture, biodiversity education.

## 4. Objectives **[Rubric 2]**
1. Acquire and prepare a public flower dataset. 2. Build a CNN using transfer learning. 3. Evaluate it on unseen data with accuracy, precision, recall, F1 and a confusion matrix.
4. Adjust parameters using evaluation results. 5. Save the model reproducibly. 6. Deploy it as a Streamlit web app. 7. Discuss responsible use.

## 5. Dataset description **[Rubric 3]**
Kaggle *Flowers Recognition* (`alxmamaev`), https://www.kaggle.com/datasets/alxmamaev/flowers-recognition. Five classes, photographs from web sources, roughly 320x240 px, uneven quality.
Public write-ups disagree on the exact size (4,242 / 4,317 / 4,323), so the exact count after cleaning is taken from `reports/evaluation/dataset_summary.json`:
total after cleaning **‹fill›**; per class **‹fill›**. Class balance: mildly unbalanced (largest class about 1.4x the smallest in public counts); the code applies class weights automatically only if the ratio exceeds 1.5.
Check the dataset's licence on Kaggle; images are not redistributed in the repository.

## 6. Data acquisition **[Rubric 3]**
`python -m training.download_dataset` uses the official `kagglehub` helper (credentials, if needed, are in the user's `.kaggle` folder - never in the code). A manual download route is documented as a fallback. Data lands in `data/raw/` (Git-ignored).

## 7. Data preprocessing **[Rubric 4]**
| Step | What | Why | Code |
|---|---|---|---|
| Discovery | find class folders even if nested | robust to Kaggle zip layout | `data.find_dataset_root` |
| Validation | fully decode every file, drop corrupted/tiny | prevent training crashes and noise | `data.is_valid_image` |
| De-duplication | MD5 of file bytes; drop copies; drop identical images filed under different classes | prevents leakage and label noise | `data.clean_images` |
| Splitting | stratified 70/15/15 on files, seed 42, leakage assertion | fair evaluation, class proportions preserved | `data.stratified_split` |
| Decoding/resizing | RGB, 224x224 bilinear | input size of the pretrained network | `preprocessing.decode_and_resize` |
| Scaling | pixels to [-1, 1] inside the model | exactly what MobileNetV2 was trained with; travels with the model file | `model.build_model` |
| Batching/prefetching | `tf.data`, batch 32, prefetch | keeps the CPU busy, efficient | `preprocessing.make_dataset` |
| Seeds | Python/NumPy/TensorFlow seeded | reproducibility | `utils.set_seed` |

**Data leakage.** Splitting happens before anything else; duplicates are removed first; a check aborts if any file or image content appears in two splits; augmentation exists only as model layers that are inactive for validation/test data, so the test set is never augmented.

## 8. Feature engineering **[Rubric 5, 6]**
*Data features:* raw pixels are converted to normalised float tensors; augmentation (horizontal flip, rotation up to +/-18 degrees, 15 % zoom, 10 % shift, brightness and contrast +/-15 %) creates varied training examples so the network becomes robust to camera angle and lighting. Vertical flips and strong colour shifts are deliberately excluded because they could change how a flower looks.
*Model features:* instead of hand-crafting features (colour histograms, edges), the pretrained convolutional base **learns** a hierarchy of features: edges -> textures -> petal/leaf parts -> object shapes. Global average pooling condenses the final 7x7x1280 feature maps into a 1280-number feature vector per image; the classification head maps it to five class probabilities.

## 9. CNN / transfer-learning methodology **[Rubric 7]**
A *convolutional neural network* slides small learnable filters over an image, detecting local patterns; stacked layers combine them into more abstract features; pooling reduces size. *Transfer learning* reuses a network trained on a large dataset (ImageNet, 1.4 M images) for a new task with little data. Training has two phases:
1. **Head training** - backbone frozen (its weights are not updated), only the new dense layer learns. Fast and safe.
2. **Fine-tuning** - upper backbone layers (index >= 100 of 154) are unfrozen and trained with a 100x smaller learning rate, adapting high-level features to flowers without destroying the general low-level ones. BatchNorm layers stay frozen for stability on a small dataset.

## 10. Model architecture **[Rubric 7]**
`Input 224x224x3 -> Augmentation -> Rescaling[-1,1] -> MobileNetV2 (ImageNet) -> GlobalAveragePooling2D -> Dropout(0.3) -> Dense(5, softmax)` (`src/model.py`; diagrams in `docs/ARCHITECTURE.md`).
*Why MobileNetV2:* ~2.3 M parameters, small saved file (fits GitHub), fast CPU inference, quick Streamlit start-up, well-supported in Keras. *Alternatives considered:* ResNet50 (~25 M parameters, larger/slower), EfficientNetB0 (accurate but slightly more finicky preprocessing and slower on CPU). The best-accuracy claim is *not* made; the choice is a balance of size, speed, reproducibility and deployment ease.

## 11. Hyper-parameter selection **[Rubric 8]**
| Parameter | Value | Explanation |
|---|---|---|
| Image size | 224x224 | native input size of the pretrained weights |
| Batch size | 32 | stable gradients, fits laptop memory |
| Split | 70/15/15 | enough training data, separate validation for tuning, separate test for the final verdict |
| Optimizer | Adam | adaptive learning rates, robust default |
| Loss | sparse categorical cross-entropy | multi-class classification with integer labels |
| Phase 1 LR | 1e-3 | new head starts from random weights and can learn fast |
| Phase 2 LR | 1e-5 | tiny steps to avoid destroying pretrained features |
| Fine-tune from layer | 100 | only high-level layers adapt |
| Dropout | 0.3 | regularisation against over-fitting |
| Epochs | up to 15 + 10 | upper bounds; early stopping decides |
| EarlyStopping | patience 4 on val loss, restore best | stops over-fitting |
| ReduceLROnPlateau | factor 0.5, patience 2 | smaller steps when progress stalls |
| Confidence threshold | 0.60 | below this the app admits uncertainty (adjustable in the UI) |
All values live in `src/config.py`.

## 12. Training process
`python -m training.train`: build datasets -> class weights if needed -> phase 1 -> validate -> phase 2 -> validate -> keep the better weights (by validation accuracy) -> save `models/flower_classifier.keras`, `class_names.json`, `config.json`, `training_history.json`, curves.
Epochs actually run: **‹fill from `model_selection.json`›**; training time: **‹fill›**.

## 13. Evaluation methodology **[Rubric 10]**
Accuracy (overall correctness), precision (of predicted X, how many are X), recall (of real X, how many were found), F1 (their harmonic mean), computed per class and averaged (macro = each class equal; weighted = by class size), and the confusion matrix (rows true, columns predicted). Learning curves (training vs validation accuracy/loss) diagnose over-/under-fitting. Final numbers come from the **test split**, evaluated once.

## 14. Results **[Rubric 11]**
| Metric | Value |
|---|---|
| Test accuracy | ‹fill› |
| Macro precision / recall / F1 | ‹fill› |
| Best / worst class (F1) | ‹fill› |
| Most frequent confusion | ‹fill› (`top_confusions`) |
*Interpretation guide:* if training accuracy >> validation accuracy -> over-fitting (more augmentation/dropout, less fine-tuning). If both are low -> under-fitting (fine-tune more layers, train longer). Confused pairs usually share colour or shape (e.g. daisy vs dandelion); look at `test_misclassified.csv` to see real examples and comment on them.

## 15. Model improvement **[Rubric 12]**
Procedure: (1) inspect phase-1 validation results and the curves; (2) run `python -m training.experiments` (A head-only, B fine-tune@100 dropout 0.3, C fine-tune@60 dropout 0.4); (3) pick the best **validation** result; (4) update `src/config.py`; (5) retrain; (6) evaluate on test once.
| Stage | Val accuracy | Val macro-F1 | Change made and reason |
|---|---|---|---|
| Initial (phase 1) | ‹fill› | ‹fill› | frozen backbone, lr 1e-3 |
| After fine-tuning (phase 2) | ‹fill› | ‹fill› | unfroze top layers, lr 1e-5 |
| Best experiment | ‹fill› | ‹fill› | ‹fill from experiments.csv› |
Do not claim an improvement that the numbers do not show; if fine-tuning did not help, say so - the code automatically keeps the better weights.

## 16. System architecture
See `docs/ARCHITECTURE.md` (Mermaid; export at mermaid.live for your document): inference flow (User -> Streamlit -> validation -> preprocessing -> CNN -> softmax -> result) and the training pipeline.

## 17. Application implementation **[Rubric 9]**
`app.py` (Streamlit): upload, preview, prediction, confidence, top-N bars, configurable uncertainty threshold, quality warnings, friendly errors for invalid/corrupt/tiny images and a missing model, cached model loading (`st.cache_resource`), a performance tab reading saved reports, and a responsible-use tab. `src/predict.py` holds the logic so it can be unit-tested (`tests/`).

## 18. Deployment **[Rubric 13, 14]**
Repository -> GitHub -> Streamlit Community Cloud (main file `app.py`). The trained model (~10 MB expected) is committed, so nothing is retrained at start-up. Paths are relative (`pathlib`), no secrets are needed. Exact steps: README §17-18.

## 19. Responsible AI
Educational tool, not a botanical authority; only 5 classes; dataset from web photos may be biased (regions, varieties, lighting, framing); accuracy depends on image quality; confidence does not guarantee correctness; wrong predictions are possible and the UI warns on low confidence; uploads are processed in memory and not stored. **Risk:** treating a prediction as fact (e.g. deciding a plant is safe to eat). **Reduction:** clear disclaimers, uncertainty threshold, quality warnings, per-class evaluation, and recommending expert verification.

## 20. Limitations
Limited classes; out-of-distribution inputs (non-flowers) are still assigned a class; single-flower assumption; moderate resolution training data; results depend on the random split and on CPU-run randomness; no calibration of probabilities.

## 21. Future work
More classes/datasets (e.g. Oxford 102), "unknown" class or out-of-distribution detection, probability calibration, Grad-CAM explanations, TensorFlow Lite, mobile camera input, larger backbones if accuracy matters more than size.

## 22. Conclusion
The project demonstrates the full deep-learning life cycle: data acquisition and cleaning, leak-free splitting, feature learning with a pretrained CNN, two-phase training, honest evaluation, evidence-based tuning, reproducible saving and web deployment. Final conclusion about achieved performance: **‹write after your results›**.

## References
Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR.
Kaggle. *Flowers Recognition* dataset (alxmamaev). https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
Keras Applications documentation; TensorFlow documentation; Streamlit documentation.
