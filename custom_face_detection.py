import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_faces(frame):
    if frame is None or frame.size == 0:
        raise ValueError("Frame inválido fornecido")
    
    # Redimensiona o frame para 400x400 para processamento consistente
    frame = cv2.resize(frame, (400, 400))
    
    try:
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                           int(bboxC.width * w), int(bboxC.height * h)
                    
                    # Garante que as coordenadas estejam dentro dos limites do frame
                    x, y, width, height = bbox
                    x = max(0, min(x, w))
                    y = max(0, min(y, h))
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    except Exception as e:
        print(f"Erro na detecção facial: {str(e)}")
        return frame
        
    return frame
