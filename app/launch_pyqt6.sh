#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname "$0")"

echo "🌊 Neptune - Surveillance Aquatique PyQt6 (Ultra Sécurisé) 🌊"
echo "============================================================="
echo

# 1) Activer conda/env
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

conda activate neptune >/dev/null 2>&1 || {
  echo "⚠️  Impossible d'activer l'env conda 'neptune'"; 
}

echo "✅ Environnement neptune activé"
echo "🚀 Lancement de Neptune PyQt6 (ultra sécurisé)..."
echo

# 2) Durcir Qt pour éviter le crash du dialogue de fichiers GTK/gdk-pixbuf
export QT_NO_XDG_DESKTOP_PORTAL=1
export QT_QPA_PLATFORM=xcb
export QT_STYLE_OVERRIDE=Fusion
export QT_ICON_THEME=hicolor

# 3) Forcer la libstdc++ système si nécessaire (évite le conflit GLIBCXX)
if [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
fi

# 4) Neutraliser le thème d'icônes problématique
export XDG_DATA_DIRS=""
export GTK_THEME=""

# 5) Forcer l'utilisation des libs système pour GTK
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# 5) Lancer l'app
python3 app.py

echo
echo "👋 Merci d'avoir utilisé Neptune!"
