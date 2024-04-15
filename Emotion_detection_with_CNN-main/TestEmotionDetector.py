import cv2
import numpy as np
from keras.models import model_from_json
import time

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the emotion detection model from JSON and weights files
with open('model/emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Define the duration for emotion detection (in seconds)
detection_duration = 5

# Start the webcam feed or use a video file
cap = cv2.VideoCapture(0)

# Lists to hold detected emotions
positive_emotions = ["Happy", "Surprised", "Neutral"]
negative_emotions = ["Angry", "Disgusted", "Fearful", "Sad"]
detected_emotions = []

# Get the current time
start_time = time.time()

print(f"Detection start time: {start_time}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detector
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each face detected
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        detected_emotion = emotion_dict[maxindex]
        detected_emotions.append(detected_emotion)

        cv2.putText(frame, detected_emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)

    # Check if 5 seconds have elapsed
    current_time = time.time()
    elapsed_time = current_time - start_time
    print(f"Current time: {current_time}, elapsed time: {elapsed_time}")

    if elapsed_time >= detection_duration:
        break

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Provide summary of emotions detected
positive_count = sum(1 for e in detected_emotions if e in positive_emotions)
negative_count = sum(1 for e in detected_emotions if e in negative_emotions)

if positive_count > negative_count:
    print("Positive review: The user expressed positive emotions more frequently.")
elif negative_count > positive_count:
    print("Negative review: The user expressed negative emotions more frequently.")
else:
    print("Neutral review: The user expressed balanced emotions.")
