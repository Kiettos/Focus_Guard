import cv2
from collections import deque

import config
from features import extract_features
from model_inference import load_model, predict
from notifier import get_notification, show_notification

def main():
    model = load_model(config.MODEL_PATH)
    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=config.WINDOW_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        features = extract_features(frame)
        if features is not None:
            buffer.append(features)

        if len(buffer) == config.WINDOW_SIZE:
            predictions = predict(model, buffer)
            message = get_notification(predictions, threshold=config.THRESHOLD)
            if message:
                show_notification(message)
            buffer.clear()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
    main()