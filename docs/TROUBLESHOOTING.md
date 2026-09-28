# Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `pip install tensorflow-cpu` finds no matching distribution | Python version too new/old for the available wheels, or 32-bit Python | Use 64-bit Python 3.12; create a fresh `.venv` with it (PyCharm -> Interpreter -> Add). |
| `ImportError: DLL load failed` when importing TensorFlow | Missing Microsoft Visual C++ Redistributable | Install the "Microsoft Visual C++ Redistributable 2015-2022 (x64)" from Microsoft, restart PyCharm. |
| NumPy errors (`numpy.core.multiarray failed to import`, ABI/"compiled with NumPy 1.x") | Mixed NumPy versions | `python -m pip install --upgrade --force-reinstall tensorflow-cpu numpy` inside the venv. |
| Pillow errors (`cannot identify image file`) | Corrupted image | Expected for bad uploads (app shows a friendly message). In training, `prepare_data` already removes such files. |
| "Could not find cuda drivers" / GPU warnings | TensorFlow looking for a GPU | Harmless. Project runs on CPU. Native Windows TensorFlow does not use the GPU anyway. |
| Training is slow | CPU-only | Normal. Try `--quick` first; reduce `phase1_epochs`/`phase2_epochs` in `src/config.py`; close other programs. |
| Out-of-memory during training | Batch too large | Lower `batch_size` to 16 in `src/config.py`. |
| `DatasetNotFoundError: Dataset folder not found` | Dataset not downloaded/extracted | Run `python -m training.download_dataset` or extract manually to `data\raw\flowers\<class>`. |
| `No class folders with images were found` | Wrong extraction layout | Class folders (`daisy`, `rose`...) must contain the `.jpg` files directly. |
| Kaggle authentication error / 401 | No credentials | Put `kaggle.json` in `C:\Users\<YOU>\.kaggle\`, or set `KAGGLE_USERNAME`/`KAGGLE_KEY`, or use the manual download. |
| Downloading ImageNet weights fails | No internet / proxy / firewall | Connect to the internet for the first training run (weights are cached afterwards). |
| `Model not found` in the app | Training not run / `models/` not committed | Run `python -m training.train`; make sure `models/flower_classifier.keras` and `class_names.json` are committed. |
| App: "The model could not be loaded" | TensorFlow/Keras version differs from the one that saved the model | Pin the exact `tensorflow-cpu==` from your venv in `requirements.txt` (README "Pin your versions") and use the same Python minor version on Streamlit Cloud. |
| `streamlit` not recognised | Not installed in this venv / venv not active | `python -m pip install -r requirements.txt`, then `python -m streamlit run app.py`. |
| `git push` rejected / "remote contains work" | Repo was created with a README | `git pull origin main --allow-unrelated-histories`, resolve, push. |
| `git push` asks for a password and fails | GitHub no longer accepts passwords | Sign in through the browser prompt (Git Credential Manager) or use a personal access token. |
| `file exceeds 100 MB` on push | Large file committed (dataset/checkpoints) | Remove from Git (`git rm --cached -r data/raw models/checkpoints`), check `.gitignore`, commit again. |
| Streamlit Cloud build fails at TensorFlow | Incompatible Python/TensorFlow combination | In *Advanced settings* select Python 3.12; try `tensorflow` instead of `tensorflow-cpu` in `requirements.txt`; read *Manage app -> logs*. |
| Streamlit Cloud app crashes/restarts (memory) | TensorFlow is heavy | Keep only one model in the app (already the case); reboot the app; avoid uploading huge images. |
| Package version conflicts | Ranges resolved differently over time | Freeze exact versions from a working venv into `requirements.txt`. |
