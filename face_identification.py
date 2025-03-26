import face_recognition
import os
import numpy as np
import cv2
from typing import List, Tuple

KNOWN_FACES_DIR = "database"
FACE_MATCH_TOLERANCE = 0.5

known_encodings: List[np.ndarray] = []
known_names: List[str] = []

def load_faces():
    if not os.path.exists(KNOWN_FACES_DIR):
        raise FileNotFoundError(f"Diretório de banco de dados '{KNOWN_FACES_DIR}' não encontrado")
        
    try:
        for name in os.listdir(KNOWN_FACES_DIR):
            if not name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(KNOWN_FACES_DIR, name)
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            
            if not encodings:
                print(f"Aviso: Nenhuma face encontrada em {name}")
                continue
                
            # Usa a primeira codificação facial se múltiplas forem encontradas
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(name)[0])
            
        if not known_encodings:
            raise ValueError("Nenhuma codificação facial válida encontrada no banco de dados")
            
    except Exception as e:
        print(f"Erro ao carregar faces: {str(e)}")
        raise

def identify_face(frame):
    if frame is None or frame.size == 0:
        raise ValueError("Frame inválido fornecido")
        
    if not known_encodings:
        raise ValueError("Nenhuma face conhecida carregada. Chame load_faces() primeiro.")
        
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if not face_locations:
            print("Nenhuma face detectada no frame")
            return frame

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                known_encodings, 
                face_encoding, 
                tolerance=FACE_MATCH_TOLERANCE
            )
            name = "Desconhecido"

            if any(matches):
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                name = "Autenticado"
                print("Rosto autenticado com sucesso")
            else:
                print("Rosto não reconhecido.")

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame
        
    except Exception as e:
        print(f"Erro na identificação facial: {str(e)}")
        return frame
