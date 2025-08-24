import cv2, time

# Load the built-in frontal face model
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("ERROR: Could not open the webcam.")

print("Press 'q' to quit.")
last_fps_time = time.time(); frames = 0; fps = 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces (tweak minSize for your camera distance)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # smaller -> more sensitive, more false positives
        minNeighbors=5,       # larger -> stricter, fewer false positives
        minSize=(80, 80)      # ignore tiny detections
    )

    # Draw boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # FPS counter (optional)
    frames += 1
    if time.time() - last_fps_time >= 1.0:
        fps = frames
        frames = 0
        last_fps_time = time.time()

    cv2.putText(frame, f"Faces: {len(faces)}  FPS: {fps}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, "Press q to quit", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
