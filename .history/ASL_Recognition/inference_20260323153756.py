import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret: break
    
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Strictly only use the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

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

        # Prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = prediction[0]

        # Draw box and text
        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
        cv2.rectangle(frame, (x1, y1), (int(max(x_) * W) + 10, int(max(y_) * H) + 10), (0, 255, 0), 2)
        cv2.putText(frame, f"Letter: {predicted_character}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    cv2.imshow('ASL Detector A-z', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()