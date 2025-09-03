#!/bin/bash

echo "🔧 Installation des dépendances pour Neptune..."

# Activer l'environnement
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neptune

echo "📦 Installation des bibliothèques IA..."

# Installer transformers et torch
pip install transformers torch torchvision torchaudio

# Installer ultralytics pour YOLO
pip install ultralytics

# Installer les autres dépendances si manquantes
pip install opencv-python numpy

echo "✅ Installation terminée!"
echo "🧪 Test d'import..."

python -c "
try:
    from transformers import AutoImageProcessor, DFineForObjectDetection
    from ultralytics import YOLO
    print('✅ Tous les imports IA sont OK!')
except ImportError as e:
    print(f'❌ Erreur d\'import: {e}')
"
