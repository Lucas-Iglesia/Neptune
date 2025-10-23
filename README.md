## Neptune (app)

This document explains how to install dependencies and launch the Neptune desktop application located in the `app/` folder.

### Prerequisites
- Python 3.11+ (the project was developed and tested with Python 3.11). If you use a different Python version, some binary wheels (PyQt6, torch) may require different installs.
- A working C/C++ build toolchain is not required for the listed wheels, but some platforms may need platform-specific packages (e.g. GPU-enabled PyTorch).

### Recommended setup (virtual environment)
1. Open a terminal and change to the app directory:

```bash
cd app
```

2. Create and activate a virtual environment (zsh / bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- The `requirements.txt` in `app/` includes: PyQt6, opencv-python, numpy, torch, transformers, ultralytics, gTTS, pygame, pyinstaller.
- If you have a CUDA-capable GPU and want GPU-accelerated PyTorch, follow the official PyTorch install instructions at https://pytorch.org/ to pick the correct CUDA wheel.

### Launching the application
From the activated virtual environment and inside the `app/` folder, run:

```bash
python3 neptune_app.py
```

The app will print a brief configuration summary and open the Qt GUI.

The app will attempt to generate audio files at startup if audio support is available (this is automatic when the `gTTS` and `pygame` dependencies are present).

### Packaging
The project includes `pyinstaller` in the requirements. To create a standalone binary, use `pyinstaller` from the activated virtual environment and inspect `neptune_app.py` as the entrypoint. Packaging details (icons, data files, hooks) are not included here and may need additional configuration.

### Troubleshooting
- Import errors for `PyQt6` or `opencv-python`: ensure the virtual environment is active and `pip install -r requirements.txt` completed without errors.
- `torch` installation errors: use the official PyTorch install helper (https://pytorch.org/get-started/locally/) to select the correct wheel for your OS and CUDA setup.
- If audio doesn't play: check that `pygame` installed correctly and that your system audio devices are available. The app only attempts to generate audio if audio dependencies are detected.

### Where to look in the code
- Entry point: `neptune_app.py`
- Main window / GUI: `ui/main_window.py`
- Core logic and video processing: `core/` folder
- Detection models: `detection/` and `model/` (contains `nwd-v2.pt`)
- Utilities for audio/alerts: `utils/`
