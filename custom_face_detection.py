import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Cache para o detector de faces
face_detection = None

def get_face_detector():
    global face_detection
    if face_detection is None:
        face_detection = mp_face_detection.FaceDetection(
            min_detection_confidence=0.7,  # Aumentado para maior precisão
            model_selection=0  # 0 para rostos próximos
        )
    return face_detection

def detect_faces(frame):
    if frame is None or frame.size == 0:
        raise ValueError("Frame inválido fornecido")
    
    # Mantém a resolução original do frame
    original_size = frame.shape[:2]
    
    try:
        # Usa o detector em cache
        face_detection = get_face_detector()
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
                
                # Desenha o retângulo
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Adiciona texto de confiança
                confidence = detection.score[0]
                cv2.putText(frame, f"Conf: {confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
    except Exception as e:
        print(f"Erro na detecção facial: {str(e)}")
        
    return frame
