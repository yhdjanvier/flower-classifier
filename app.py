"""
Flower Classifier - Streamlit web application.

Run locally:   streamlit run app.py
Deployed on:   Streamlit Community Cloud (main file = app.py)

Flow: upload image -> validate -> pre-process -> MobileNetV2 CNN -> softmax ->
      top-N predictions + honest "uncertain" warning when confidence is low.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    ROOT = Path(__file__).resolve().parent
except NameError:  # e.g. some test runners
    ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src.predict import (  # noqa: E402
    FlowerPredictor,
    InvalidImageError,
    ModelNotFoundError,
    PredictionError,
    load_image,
)
from src.utils import download_file, load_json  # noqa: E402

log = logging.getLogger("flower-app")

st.set_page_config(page_title="Flower Classifier", page_icon="🌸", layout="centered")

EMOJI = {"daisy": "🌼", "dandelion": "🌻", "rose": "🌹", "sunflower": "🌻", "tulip": "🌷"}
FIG, EVAL_DIR = C.FIGURES_DIR, C.EVAL_DIR


# ----------------------------------------------------------------------------- model loading (cached: once per server)
def _optional_model_download() -> None:
    """Optional fallback: if the model file is not in the repository, download it from a
    public HTTPS URL given as the MODEL_URL secret / environment variable."""
    if C.MODEL_FILE.exists():
        return
    url = os.environ.get("MODEL_URL")
    if not url:
        try:
            url = st.secrets.get("MODEL_URL")
        except Exception:  # no secrets file configured
            url = None
    if url:
        download_file(url, C.MODEL_FILE)


@st.cache_resource(show_spinner="Loading the CNN model (first time only)...")
def get_predictor() -> FlowerPredictor:
    _optional_model_download()
    return FlowerPredictor.from_directory(C.MODELS_DIR)


try:
    predictor = get_predictor()
    load_problem = None
except ModelNotFoundError as exc:
    predictor, load_problem = None, str(exc)
except Exception as exc:  # corrupted file, incompatible Keras version, ...
    log.exception("Model loading failed")
    predictor, load_problem = None, "The model could not be loaded. Please try again later or contact the app owner."


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    default_threshold = predictor.confidence_threshold if predictor else C.Config().confidence_threshold
    threshold = st.slider("Confidence threshold", 0.30, 0.95, float(default_threshold), 0.05,
                          help="Below this confidence the app says the model is uncertain.")
    top_n = st.slider("Predictions to show", 2, 5, 3)
    st.caption("Educational demo - not a botanical identification tool.")
    if predictor:
        st.caption("Classes: " + ", ".join(predictor.class_names))

st.title("🌸 Flower Classifier")
st.write("Upload a photo of a flower. A convolutional neural network (MobileNetV2, transfer learning) predicts its type.")

if load_problem:
    st.error(load_problem)
    st.info("Developer note: train the model with `python -m training.train`, or commit `models/flower_classifier.keras`.")
    st.stop()

tab_classify, tab_perf, tab_about = st.tabs(["Classify", "Model performance", "About & responsible use"])

# ----------------------------------------------------------------------------- tab 1: classify
with tab_classify:
    upload = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png", "bmp", "webp"],
                              help="One image, JPG/PNG/BMP/WEBP, ideally the flower clearly in the centre.")
    if upload is None:
        st.info("👆 Choose an image to begin.")
    else:
        try:
            image = load_image(upload)
        except InvalidImageError as exc:
            st.error(str(exc))
        else:
            result = None
            left, right = st.columns([1, 1])
            with left:
                st.image(image, caption=f"Uploaded image ({image.size[0]}x{image.size[1]})")
            with right:
                try:
                    with st.spinner("Analysing image..."):
                        result = predictor.predict(image, top_k=top_n, confidence_threshold=threshold)
                except PredictionError as exc:
                    st.error(str(exc))
                except Exception:  # never show a traceback to users
                    log.exception("Unexpected prediction failure")
                    st.error("Something went wrong while analysing the image. Please try a different one.")
                else:
                    icon = EMOJI.get(result.label, "🌸")
                    if result.is_uncertain:
                        st.warning(f"Best guess: **{icon} {result.label.title()}**, but the model is uncertain. "
                                   "Please try a clearer image.")
                    else:
                        st.success(f"Prediction: **{icon} {result.label.title()}**")
                    st.metric("Confidence", f"{result.confidence:.2%}")
                    st.caption("Confidence is the model's own probability estimate. It does not guarantee the answer is correct.")
                    for note in result.warnings:
                        st.warning(note)
            if result is not None:
                st.subheader("Top predictions")
                for name, prob in result.top_k:
                    st.progress(min(max(prob, 0.0), 1.0), text=f"{name.title()} - {prob:.2%}")
                with st.expander("What happened behind the scenes?"):
                    st.markdown(
                        "1. Image validated (format, size, decodable)\n"
                        f"2. Resized to {predictor.img_size[0]}x{predictor.img_size[1]} pixels\n"
                        "3. Pixels scaled to [-1, 1] and passed through MobileNetV2 (learned visual features)\n"
                        "4. Classification layer + softmax turn features into class probabilities\n"
                        "5. The highest probability is the prediction; low values trigger the 'uncertain' message"
                    )

# ----------------------------------------------------------------------------- tab 2: model performance
with tab_perf:
    metrics_file = EVAL_DIR / "test_metrics.json"
    if not metrics_file.exists():
        st.info("No evaluation results found yet. Run `python -m training.evaluate` and commit `reports/`.")
    else:
        m = load_json(metrics_file)
        st.caption(f"Computed on {m['n_samples']} held-out test images the model never saw during training.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy']:.2%}")
        c2.metric("Macro precision", f"{m['macro_precision']:.2%}")
        c3.metric("Macro recall", f"{m['macro_recall']:.2%}")
        c4.metric("Macro F1", f"{m['macro_f1']:.2%}")
        st.subheader("Per-class results")
        per_class = pd.DataFrame(m["per_class"]).T.rename_axis("class").reset_index()
        st.dataframe(per_class)
        if m.get("top_confusions"):
            st.caption("Most frequent mistakes: " + "; ".join(
                f"{e['true']} -> {e['predicted']} (x{e['count']})" for e in m["top_confusions"]))
        for title, file in (("Confusion matrix", FIG / "test_confusion_matrix.png"),
                            ("Training curves", FIG / "training_curves.png")):
            if file.exists():
                st.subheader(title)
                st.image(str(file))
    sel = EVAL_DIR / "model_selection.json"
    if sel.exists():
        s = load_json(sel)
        st.subheader("Model improvement (validation set)")
        rows = [{"stage": "Phase 1 - head only", **s["val_phase1"]}]
        if s.get("val_phase2"):
            rows.append({"stage": "Phase 2 - fine-tuned", **s["val_phase2"]})
        st.table(pd.DataFrame(rows).set_index("stage").round(4))
        st.caption(f"Weights kept for the final model: {s['chosen']}.")

# ----------------------------------------------------------------------------- tab 3: about
with tab_about:
    st.markdown(
        """
**How it works.** A MobileNetV2 convolutional network pre-trained on ImageNet is used as a feature extractor,
and a small classification head was trained on the Kaggle *Flowers Recognition* dataset (5 flower types).

**Responsible use**
- ⚠️ Educational demo, **not** a definitive botanical identification system. Never use it for decisions about
  edible/poisonous plants, medicine or safety.
- The model only knows **5 classes**. A different flower (or a non-flower) will still be forced into one of them.
- Accuracy depends on image quality: blurry, dark, distant or cluttered photos are harder.
- The training photos come from web sources and may under-represent some regions, varieties and lighting conditions (dataset bias).
- A high confidence does **not** guarantee a correct answer. Treat results as suggestions.
- Uploaded images are analysed in memory and are not saved by this app.
        """
    )
