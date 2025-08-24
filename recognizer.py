import cv2
import numpy as np
import os

# Load trained model + names
trainer_path = "trainer"
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(trainer_path, "trainer.yml"))
names = np.load(os.path.join(trainer_path, "names.npy"), allow_pickle=True).item()

# Load Haar cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("ERROR: Could not open the webcam.")

print("Press 'q' to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        # Predict
        label, confidence = recognizer.predict(face_img)

        if confidence < 70:  # lower confidence = better match
            name = names[label]
            text = f"{name} ({round(100 - confidence)}%)"
            color = (0, 255, 0)
        else:
            text = "Unknown"
            color = (0, 0, 255)

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
