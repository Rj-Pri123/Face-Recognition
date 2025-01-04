import cv2
from fer import FER

emotion_detector = FER()

cap = cv2.VideoCapture(0)

emotion_colors = {
    'happy': (0, 255, 0),  # Green for happy
    'neutral': (255, 255, 0),  # Yellow for neutral
    'sad': (255, 0, 0),  # Red for sad
    'angry': (0, 0, 255),  # Blue for angry
    'disgust': (255, 0, 255),  # Purple for disgust
    'fear': (0, 255, 255),  # Cyan for fear
    'surprise': (255, 255, 255)  # White for surprise
}


def draw_txt(frame, text, x, y, font_scale=1, thickness=2):
    for i, line in enumerate(text.split('\n')):
        cv2.putText(frame, line, (x, y + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness,
                    cv2.LINE_AA)


while True:
    ret, frame = cap.read()  # Capture a frame
    if not ret:
        print('Error: Failed to capture image.')
        break

    emotion_result = emotion_detector.detect_emotions(frame)

    if emotion_result:
        for face in emotion_result:
            (x, y, w, h) = face['box']

            dominant_emotion = max(face['emotions'], key=face['emotions'].get)
            confidence = face['emotions'][dominant_emotion] * 100

            color = emotion_colors.get(dominant_emotion, (255, 255, 255))

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


            if dominant_emotion == "happy" and confidence > 50:
                text = "Your smile is Most \nBeautiful in the world!"
                draw_txt(frame, text, x, y - 30)
            else:

                text = "Please keep a smile \non your Most Beautiful face!"
                draw_txt(frame, text, x, y - 30)

            cv2.putText(frame, f"{dominant_emotion.upper()} ({confidence:.2f}%)", (x, y + h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
