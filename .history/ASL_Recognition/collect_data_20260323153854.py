import cv2
import os

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Updated for A through G
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H','I','J','K','L','M','N','O','P','Q','R','S'] 
dataset_size = 100

cap = cv2.VideoCapture(0)
for j in alphabet:
    if not os.path.exists(os.path.join(DATA_DIR, j)):
        os.makedirs(os.path.join(DATA_DIR, j))

    print(f'Collecting data for letter: {j}')

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Sign "{j}" and press "Q"', (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, j, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()