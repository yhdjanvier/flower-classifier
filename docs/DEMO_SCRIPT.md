# Demonstration procedure (about 8-10 minutes)

## Before the demo (10 minutes earlier)
- `streamlit run app.py` works locally; the public Streamlit URL loads (open it once - free apps sleep when idle).
- Have 4 photos ready: a clear flower of each of 2-3 classes, one **blurry/odd** photo, one **non-flower** (e.g. a car), one non-image file (e.g. `.txt`).
- Have `reports/figures/test_confusion_matrix.png` and the GitHub repo open in tabs.

## Steps and what to say
1. **Start** - "This is a CNN-based flower classifier deployed with Streamlit." Show the terminal command `streamlit run app.py`.
2. **Explain the app** - "Upload a photo -> the app validates it -> the CNN predicts one of 5 flowers with a confidence."
3. **Upload a clear flower** - Show the preview, prediction, confidence and the top-3 bars.
4. **Explain preprocessing** - Open "What happened behind the scenes": resize to 224x224, scale to [-1,1]. "Training uses the same steps, so results are consistent."
5. **Explain the model** - MobileNetV2 pretrained on ImageNet = feature extractor; global pooling; dropout; softmax head. "Transfer learning: reuse learned edges/textures/shapes, train only what is specific to flowers."
6. **Show evaluation** - Tab *Model performance*: accuracy, precision, recall, F1 (define each in one sentence).
7. **Confusion matrix** - "Rows are true classes, columns predictions, the diagonal is correct; the largest off-diagonal cell shows the most confused pair - [name them from your own results] - and why (similar colour/shape)."
8. **Model improvement** - Show phase 1 vs phase 2 table and `experiments.csv`: what you changed and what the validation results said.
9. **Robustness** - Upload the blurry photo (quality warning / uncertain message), the non-flower (forced into a class - explain why), the `.txt` (friendly error).
10. **GitHub** - Show the repository: structure, `.gitignore`, no dataset/secrets, model committed.
11. **Deployed app** - Open the public URL and classify one image.
12. **Responsible AI** - Educational only; 5 classes; dataset bias; image-quality dependence; confidence != correctness.

## Short spoken script
"My project classifies flower photos using a convolutional neural network. I used the Kaggle Flowers Recognition dataset with five classes.
I cleaned it, removed duplicates, and split it 70/15/15 by file so no image appears in two sets. Because the dataset is small, I used transfer learning:
MobileNetV2 pretrained on ImageNet acts as a learned feature extractor, and I train a small head on top - first with the backbone frozen, then fine-tuning
the top layers with a much smaller learning rate. Augmentation like flips and small rotations makes the model more robust and is applied only during training.
I evaluated on a test set the model never saw, using accuracy, precision, recall, F1 and a confusion matrix, and I used validation results - not the test set -
to choose parameters. The model is saved with its class names and configuration, and the Streamlit app loads it once and shows predictions with confidence,
warning the user when the model is uncertain. The app is deployed from GitHub to Streamlit Cloud. It is an educational tool, not a botanical authority."

## Manual test checklist
- [ ] App opens; sidebar shows the class list; no error banner.
- [ ] No upload -> friendly "choose an image" message.
- [ ] JPG, PNG and WEBP flower photos each work.
- [ ] Prediction, confidence %, and top-N bars shown; percentages sum to about 100 %.
- [ ] Low threshold/high threshold slider changes the "uncertain" behaviour.
- [ ] A `.txt` renamed to `.jpg` -> friendly error, no traceback.
- [ ] A tiny image (< 64 px) -> friendly error.
- [ ] Blank/very dark image -> warning shown.
- [ ] Very large image (a few MB) still works.
- [ ] Non-flower image -> still returns a class but typically lower confidence; explain limitation.
- [ ] *Model performance* tab shows your real metrics and figures.
- [ ] Renaming `models/flower_classifier.keras` temporarily -> clear "model not found" message (then restore it).
- [ ] Public Streamlit URL behaves the same as local.
