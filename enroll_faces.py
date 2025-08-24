import cv2, os

# Load the Haar Cascade face detector
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Ask for the person's name
person_name = input("Enter the person's name: ").strip()
save_dir = os.path.join("dataset", person_name)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("ERROR: Could not open the webcam.")

print("Collecting face samples. Press 'q' to stop early.")
count = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

    for (x, y, w, h) in faces:
        # Crop and save face
        face_img = gray[y:y+h, x:x+w]
        count += 1
        file_path = os.path.join(save_dir, f"{count}.jpg")
        cv2.imwrite(file_path, face_img)

        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Saved: {count}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Enrolling Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count >= 50:   # stop after 50 images
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Saved {count} images to {save_dir}")

