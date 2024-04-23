# after running this, it will create the trained_model.yml for you
import os
import cv2
import numpy as np

# Path to the dataset
data_path = 'data_set'  # Correct path to your dataset

# Create a face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prepare training data
def get_images_and_labels(path):
    labels = []
    faces = []
    ids = []
    label_id = 0
    label_map = {}

    # Iterate through each person's folder
    for label_name in os.listdir(path):
        person_path = os.path.join(path, label_name)
        if not os.path.isdir(person_path):
            continue

        # Process each image in the person's folder
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            faces_detected = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
            print(f"Faces detected in {image_path}: {len(faces_detected)}")

            for (x, y, w, h) in faces_detected:
                faces.append(image[y:y+h, x:x+w])
                labels.append(label_id)
                print(f"Data added for label {label_id}")

        label_map[label_id] = label_name
        label_id += 1

    return faces, labels, label_map

faces, labels, label_map = get_images_and_labels(data_path)
print(f"Data Prepared: {len(faces)} faces, {len(labels)} labels from {len(label_map)} persons")

# Check if enough data is collected
if len(faces) > 1 and len(set(labels)) > 1:
    # Train the recognizer only if there are enough faces and at least two different labels
    recognizer.train(faces, np.array(labels))
    print("Model trained successfully")
    recognizer.save('trained_model.yml')
    print("Model saved as 'trained_model.yml'")
else:
    print("Not enough data to train the model.")
