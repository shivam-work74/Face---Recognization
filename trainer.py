import cv2, os
import numpy as np

# Path to dataset
dataset_path = "dataset"
trainer_path = "trainer"
os.makedirs(trainer_path, exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascade_path)

faces = []
labels = []
names = {}
current_id = 0

print("Training faces. Please wait...")

# Loop through dataset folders
for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir):
        continue

    names[current_id] = person_name  # map label id to name

    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if gray is None:
            continue

        faces.append(gray)
        labels.append(current_id)

    current_id += 1

# Train recognizer
recognizer.train(faces, np.array(labels))

# Save model + label mapping
recognizer.save(os.path.join(trainer_path, "trainer.yml"))
np.save(os.path.join(trainer_path, "names.npy"), names)

print(f"✅ Training complete. Model saved in '{trainer_path}/'")
