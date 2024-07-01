from time import sleep

import cv2
import numpy as np
import mediapipe as mp

# Load MediaPipe object detection model
mp_traffic_signs = mp.solutions.ObjectDetection(model_name='traffic_signs')

def preprocess_image(image):
    # MediaPipe expects RGB image, but OpenCV captures BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def detect_traffic_sign(image):
    results = mp_traffic_signs.process(image)
    if results.detections:
        detection = results.detections[0]  # assume only one detection
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        ymin, xmin, ymax, xmax = bboxC.ymin, bboxC.xmin, bboxC.ymax, bboxC.xmax
        (left, right, top, bottom) = (int(xmin * iw), int(xmax * iw), int(ymin * ih), int(ymax * ih))
        sign = image[top:bottom, left:right]
        return True, sign
    else:
        return False, None

labelToText = {
    0: "Limite de Velocidade (30km/h)",
    1: "Pare",
    2: "Não entre",
    3: "Estrada em obras",
    4: "Pedestres",
    5: "Crianças",
    6: "Bicicletas",
    7: "Animais selvagens",
    8: "Siga em frente",
    9: "Mantenha a direita",
    10: "Mantenha a esquerda"
}

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame_rgb = preprocess_image(frame)

    found, sign = detect_traffic_sign(frame_rgb)
    if found:
        cv2.imshow('frame', frame)
        cv2.imshow('sign', sign)
        # Assuming predict4() is still used for classification
        result = labelToText[predict4(sign)]
        print(result)
    else:
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()