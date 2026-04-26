import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/coffee_model_v1.h5")

#Frame preprocessor
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

#Predict function
def predict(img):
    pred = model.predict(img)
    prob = float(pred[0][0])

    label = "GOOD" if prob > 0.5 else "BAD"
    return label, prob

#Start web cam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = preprocess(frame)

    # Predict
    label, prob = predict(img)

    # Choose color
    color = (0, 255, 0) if label == "GOOD" else (0, 0, 255) # green for good and red for bad

    # Draw border
    cv2.rectangle(frame, (0, 0),
                  (frame.shape[1], frame.shape[0]),
                  color, 10)

    # Put label text
    text = f"{label} ({prob:.2f})"
    cv2.putText(frame, text, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, color, 3)

    cv2.imshow("Bean Classifier", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
