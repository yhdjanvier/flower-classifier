# Architecture diagrams (Mermaid)

GitHub renders these automatically. To put them in your Word/PDF report: open <https://mermaid.live>, paste a diagram,
click **Actions -> PNG/SVG**. In PyCharm, install the free *Mermaid* plugin to preview.

## 1. Inference (what happens when a user uploads a photo)

```mermaid
flowchart TD
    U([User]) --> S[Streamlit interface]
    S --> UP[Image upload]
    UP --> V{Image validation<br/>format, readable, size}
    V -- invalid --> ERR[Friendly error message]
    V -- valid --> P[Preprocessing<br/>EXIF fix, RGB, resize 224x224]
    P --> R[Rescale pixels to -1..1]
    R --> B[MobileNetV2 backbone<br/>feature extraction]
    B --> G[Global average pooling<br/>1280 features]
    G --> D[Dropout - inactive at inference]
    D --> H[Dense classification head]
    H --> SM[Softmax probabilities]
    SM --> PC[Prediction + confidence + top-N]
    PC --> T{confidence >= threshold?}
    T -- yes --> OK[Show flower name]
    T -- no --> UNC[Show 'uncertain' warning]
    OK --> S
    UNC --> S
```

## 2. Training pipeline

```mermaid
flowchart TD
    A[Kaggle Flowers Recognition dataset] --> B[Data acquisition<br/>download_dataset.py]
    B --> C[Cleaning<br/>corrupted files, duplicates]
    C --> D[Stratified split<br/>70 / 15 / 15, seed 42]
    D --> TR[(Train)]
    D --> VA[(Validation)]
    D --> TE[(Test - untouched until the end)]
    TR --> AUG[Preprocessing + augmentation<br/>train only]
    VA --> PRE[Preprocessing only]
    AUG --> M[Pretrained MobileNetV2]
    PRE --> M
    M --> P1[Phase 1: train head, backbone frozen]
    P1 --> P2[Phase 2: fine-tune top layers, low LR]
    P2 --> SEL[Select best weights on VALIDATION]
    SEL --> SAVE[(Saved model + class names + config)]
    SAVE --> EV[Evaluation on TEST set<br/>accuracy, P, R, F1, confusion matrix]
    SAVE --> DEP[GitHub -> Streamlit Cloud]
    TE --> EV
```

## 3. Model layers

```mermaid
flowchart LR
    I[Input 224x224x3] --> A[Augmentation<br/>training only] --> RS[Rescaling] --> MB[MobileNetV2<br/>ImageNet weights] --> GP[GlobalAvgPool] --> DO[Dropout 0.3] --> DE[Dense 5 softmax]
```
