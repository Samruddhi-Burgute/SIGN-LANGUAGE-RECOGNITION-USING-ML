import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

print("Starting training data processing...")

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        
        # Ensure we only extract 42 features (one hand)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0] 
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Only append if we successfully got exactly 42 coordinates
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)

# Create and save the model
print(f"Training model on {len(data)} samples...")
model = RandomForestClassifier()
model.fit(np.asarray(data), np.asarray(labels))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Training Complete. Model saved as model.p")