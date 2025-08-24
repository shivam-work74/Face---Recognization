import cv2, time

print("OpenCV version:", cv2.__version__)
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    raise SystemExit("ERROR: Could not open the webcam. Is it connected or used by another app?")

print("Press 'q' to quit the preview window.")
last_fps_time = time.time(); frames = 0; fps = 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed.")
        break

    # simple FPS counter
    frames += 1
    if time.time() - last_fps_time >= 1.0:
        fps = frames
        frames = 0
        last_fps_time = time.time()

    cv2.putText(frame, f"FPS: {fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Webcam Test - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
