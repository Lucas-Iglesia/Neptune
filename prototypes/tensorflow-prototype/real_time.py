import cv2
import numpy as np
import tensorflow as tf

# Charger le modèle
model = tf.keras.models.load_model('gesture_detection_model.h5')

def detect_gestures(model):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img = cv2.resize(frame, (64, 64))
            img = np.expand_dims(img, axis=0) / 255.0
            prediction = model.predict(img)

            print(prediction)

            if prediction > 0.95:
                label = 'Danger'
            else:
                label = 'Normal'

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

# Détection en temps réel
detect_gestures(model)