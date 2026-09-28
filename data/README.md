# Data folder

Nothing here is committed to Git except this file. The dataset is reproduced by scripts:

```powershell
python -m training.download_dataset   # -> data/raw/flowers/...
python -m training.prepare_data       # -> data/processed/splits.csv
```

**Dataset:** Flowers Recognition (Kaggle, author `alxmamaev`)
**URL:** https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
**Classes:** daisy, dandelion, rose, sunflower, tulip
**Images:** about 4,200-4,300 JPEGs (public write-ups quote 4,242 / 4,317 / 4,323; the exact number after cleaning is
written by `prepare_data` to `reports/evaluation/dataset_summary.json`). Roughly 320x240 px, photos of varying quality.
**Licence / usage:** check the licence shown on the Kaggle page before redistributing. Images are not stored in this
repository; use them for education/research only.

Manual route (no API key): download `archive.zip` from the URL above and extract it into `data/raw/` so that
`data/raw/flowers/daisy/*.jpg` exists.
