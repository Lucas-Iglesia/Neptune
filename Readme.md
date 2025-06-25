# Neptune

Readme to understand how to run different parts of Neptune.

---

## 🚀 Quick Start

### 1 · Install Conda (see below) or verify it’s on your PATH

```bash
conda --version
```

### 2 · Create and activate the D-FINE environment

```bash
conda create -n dfine python=3.11                # only first time
conda activate dfine
pip install -r dfine-requirements.txt            # install core deps
```

### 3 · Create and activate the YOLOv11 environment

```bash
conda create -n yolo11 python=3.11               # only first time
conda activate yolo11
pip install -r yolo11-requirements.txt           # install vision deps
```
