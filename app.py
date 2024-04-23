import cv2
import numpy as np

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.yml')

# Start capturing video from the webcam
video_capture = cv2.VideoCapture(2)  # Change to 0 for default camera

# Load reference image and convert to grayscale
reference_image = cv2.imread('trump2.jpeg')
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(reference_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

ref_label = None  # Initialize ref_label
if len(faces) > 0:
    # Assume first detected face is the person to match
    x, y, w, h = faces[0]
    reference_roi = reference_gray[y:y+h, x:x+w]
    # You might need to train your recognizer here if it doesn't already know this face
    # For simplicity, let's just assign a label if you know it, e.g., 0
    ref_label = 0  # This needs to match with your trained labels

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_roi)

        # Compare against the reference face
        if ref_label is not None and label == ref_label and confidence < 100:  # Adjust the confidence threshold as needed
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Match {confidence:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
