"""
Download the Kaggle "Flowers Recognition" dataset into data/raw/flowers.

    python -m training.download_dataset

Dataset : Flowers Recognition (alxmamaev) - https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
Method  : `kagglehub` (official Kaggle helper). If it asks for credentials, or you are
          offline, use the MANUAL route printed at the end (no code needed).
Secrets : credentials are read by kagglehub from  C:\\Users\\<YOU>\\.kaggle\\kaggle.json
          (or the KAGGLE_USERNAME / KAGGLE_KEY environment variables). They are never
          stored in this project.
"""

from __future__ import annotations

import shutil

from src import config as C
from src.utils import get_logger

log = get_logger("download")

MANUAL_HELP = f"""
MANUAL DOWNLOAD (works without any API key):
  1. Open {C.KAGGLE_URL} in your browser and sign in to Kaggle (free).
  2. Click "Download" -> you get archive.zip.
  3. Extract it so that the class folders end up here:
       {C.RAW_DIR / 'flowers' / 'daisy'}
       {C.RAW_DIR / 'flowers' / 'dandelion'}   ... and so on for rose, sunflower, tulip.
     (A nested folder such as flowers\\flowers\\daisy is also fine.)
  4. Then run:  python -m training.prepare_data
"""


def main() -> None:
    target = C.RAW_DIR / "flowers"
    if target.exists() and any(target.rglob("*.jpg")):
        log.info("Dataset already present at %s - nothing to do.", target)
        return
    try:
        import kagglehub
    except ImportError:
        log.error("kagglehub is not installed. Run: pip install -r requirements-dev.txt")
        print(MANUAL_HELP)
        return
    try:
        log.info("Downloading %s ... (about 230 MB)", C.KAGGLE_DATASET)
        cache_path = kagglehub.dataset_download(C.KAGGLE_DATASET)
    except Exception as exc:  # network, auth, or Kaggle-side problem
        log.error("Automatic download failed: %s", exc)
        print(MANUAL_HELP)
        return
    log.info("Downloaded to Kaggle cache: %s - copying into the project ...", cache_path)
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_path, target, dirs_exist_ok=True)
    log.info("Done. Next: python -m training.prepare_data")


if __name__ == "__main__":
    main()
