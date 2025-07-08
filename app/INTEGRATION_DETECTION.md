# Integration des nouvelles fonctionnalités de detection.py

## Modifications apportées

### Backend (Python Flask)

#### 1. Fichier `homography/processor.py`
- **Mise à jour de la classe `UnderwaterPersonTracker`** :
  - Ajout d'un système de callback pour les alertes
  - Modification des noms des variables pour correspondre au nouveau système
  - Ajout du suivi des alertes vocales (`voice_alert_sent`)
  - Score de danger renommé en `dangerosity_score`

- **Nouvelles fonctions ajoutées** :
  - `calculate_dangerosity_score()` : Calcule le score de danger (0-100)
  - `get_color_by_dangerosity()` : Couleur basée sur le score de danger
  - `calculate_distance_from_shore()` : Distance depuis la côte
  - `get_color_for_track()` : Couleur pour chaque track

- **Améliorations de la classe `HomographyProcessor`** :
  - Ajout d'une queue d'alertes (`alert_queue`)
  - Méthode `get_alerts()` pour récupérer les alertes
  - Callback `_on_alert()` pour recevoir les alertes du tracker

#### 2. Fichier `main.py`
- **Système d'alertes en temps réel** :
  - Thread de monitoring des alertes
  - Queue thread-safe pour les alertes
  - Modification du endpoint `/api/alerts` pour utiliser les vraies alertes
  - Ajout d'un endpoint `/api/alert-audio` pour servir l'audio

#### 3. Fichier `generate_alert_audio.py`
- Script pour générer l'audio d'alerte une seule fois
- Utilise gTTS (Google Text-to-Speech) pour créer le fichier MP3
- Message en français : "Alerte ! Baigneur en danger."

### Frontend (React/TypeScript)

#### 1. Composant `Alert.tsx`
- **Lecture audio automatique** :
  - Utilisation de `useRef` pour contrôler l'élément audio
  - Propriété `playAudio` pour déclencher la lecture
  - Gestion des erreurs audio

#### 2. Fichier `App.tsx`
- Mise à jour du type des alertes pour inclure `playAudio`
- Correction de l'utilisation de `handleDelete`

## Fonctionnalités principales

### 1. Détection sous-marine améliorée
- **Tracking sophistiqué** : Suivi des personnes entre surface et sous l'eau
- **Score de dangerosité** : Calcul en temps réel (0-100) basé sur :
  - Distance de la côte (20 points max)
  - Temps de plongée (30 points max)
  - Statut sous l'eau (20 points max)
  - Temps de danger (40 points max)
  - Excès de frames sous l'eau (10 points max)

### 2. Alertes en temps réel
- **Backend** : Détection des situations dangereuses (>5 secondes sous l'eau)
- **Communication** : Server-Sent Events (SSE) pour les alertes temps réel
- **Frontend** : Réception et affichage des alertes avec audio

### 3. Système audio
- **Génération unique** : Fichier MP3 créé une seule fois
- **Lecture automatique** : Audio joué automatiquement lors des alertes
- **Optimisation** : Pas de génération audio répétée

## Configuration

### Paramètres modifiables dans `processor.py`
```python
CONF_THRES = 0.55                    # Seuil de confiance détection
DANGER_TIME_THRESHOLD = 5            # Secondes avant alerte danger
UNDERWATER_THRESHOLD = 15            # Frames avant statut "sous l'eau"
SURFACE_THRESHOLD = 5                # Frames avant statut "surface"
UPDATE_EVERY = 30                    # Fréquence mise à jour homographie
```

## Utilisation

### 1. Génération de l'audio (une seule fois)
```bash
cd app/backend
python generate_alert_audio.py
```

### 2. Lancement du backend
```bash
cd app/backend
python main.py
```

### 3. Lancement du frontend
```bash
cd app/frontend
npm run dev
```

## Améliorations apportées

1. **Performance** : Réduction du temps de traitement par frame
2. **Précision** : Meilleur calcul du score de dangerosité
3. **User Experience** : Alertes audio instantanées
4. **Maintenabilité** : Code plus modulaire et commenté
5. **Temps réel** : Système d'alertes vraiment en temps réel

## Fonctionnalités supprimées

- **Pygame** : Toutes les dépendances pygame ont été supprimées
- **Génération audio répétée** : Un seul fichier audio généré
- **Alertes factices** : Remplacement par de vraies alertes basées sur la détection
