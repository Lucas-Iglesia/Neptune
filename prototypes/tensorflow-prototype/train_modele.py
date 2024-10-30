import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Charger le modèle VGG16 pré-entraîné sans la couche de classification
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Ajouter des couches de classification personnalisées
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Définir le modèle complet
model = Model(inputs=base_model.input, outputs=predictions)

# Geler les couches du modèle de base
for layer in base_model.layers:
    layer.trainable = False

# Compiler le modèle
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Préparer les générateurs de données avec augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'data',  # Dossier contenant les sous-dossiers 'normal_gestures' et 'danger_gestures'
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'data',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Entraîner le modèle
model.fit(train_generator, epochs=20, validation_data=validation_generator)

# Sauvegarder le modèle
model.save('gesture_detection_model.h5')