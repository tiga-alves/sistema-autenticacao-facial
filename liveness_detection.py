import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, List

# Configurações
LAPLACIAN_THRESHOLD = 30
BLINK_THRESHOLD = 0.3
MOVEMENT_THRESHOLD = 0.1
MIN_FRAMES_FOR_DETECTION = 30  # Reduzido para resposta mais rápida
INITIALIZATION_FRAMES = 15  # Reduzido para resposta mais rápida

# Inicialização do MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7
)

class LivenessDetector:
    def __init__(self):
        self.previous_frame = None
        self.previous_landmarks = None
        self.frame_count = 0
        self.blink_count = 0
        self.last_blink_state = False
        self.is_initialized = False
        self.debug_info = {}
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
    def get_eye_aspect_ratio(self, landmarks) -> float:
        LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
        
        def calculate_ear(eye_points):
            v1 = np.linalg.norm(landmarks[eye_points[1]] - landmarks[eye_points[5]])
            v2 = np.linalg.norm(landmarks[eye_points[2]] - landmarks[eye_points[4]])
            h = np.linalg.norm(landmarks[eye_points[0]] - landmarks[eye_points[3]])
            return (v1 + v2) / (2.0 * h)
        
        left_ear = calculate_ear(LEFT_EYE)
        right_ear = calculate_ear(RIGHT_EYE)
        return (left_ear + right_ear) / 2.0

    def detect_movement(self, current_landmarks, previous_landmarks) -> float:
        if previous_landmarks is None:
            return 0.0

        movement = np.mean(np.abs(current_landmarks - previous_landmarks))
    
        # Smooth movement by averaging over the last few frames
        if not hasattr(self, 'movement_buffer'):
            self.movement_buffer = []
    
        self.movement_buffer.append(movement)
        if len(self.movement_buffer) > 5:  # Keep the last 5 movements
            self.movement_buffer.pop(0)
    
        avg_movement = np.mean(self.movement_buffer)
        self.debug_info['average_movement'] = avg_movement
        return avg_movement

    def analyze_image_quality(self, frame) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Métricas de qualidade
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        uniformity = np.sum(hist ** 2)
        contrast = np.std(gray)
        
        # Armazena informações de debug
        self.debug_info.update({
            'laplacian': laplacian,
            'uniformity': uniformity,
            'contrast': contrast
        })
        
        # Verifica se é uma foto
        is_photo = (laplacian < LAPLACIAN_THRESHOLD or 
                   uniformity > 0.5 or 
                   contrast < 20 or 
                   contrast > 200)
        
        if is_photo:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
            
        return not is_photo

    def detect_liveness(self, frame) -> bool:
        if frame is None or frame.size == 0:
            raise ValueError("Frame inválido fornecido")
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                if self.frame_count < INITIALIZATION_FRAMES:
                    return True
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    self.debug_info['reason'] = 'Nenhuma face detectada'
                    return False
                return True
                
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
            
            if self.frame_count < INITIALIZATION_FRAMES:
                self.previous_landmarks = landmarks
                self.frame_count += 1
                return True
            
            # Análise de qualidade da imagem
            if not self.analyze_image_quality(frame):
                if self.consecutive_failures >= self.max_consecutive_failures:
                    self.debug_info['reason'] = 'Qualidade da imagem suspeita'
                    return False
                return True
                
            # Detecção de piscadas
            ear = self.get_eye_aspect_ratio(landmarks)
            is_blinking = ear < BLINK_THRESHOLD
            
            if is_blinking and not self.last_blink_state:
                self.blink_count += 1
                self.consecutive_failures = 0  # Reseta falhas após piscada
            self.last_blink_state = is_blinking
            
            # Detecção de movimento
            movement = self.detect_movement(landmarks, self.previous_landmarks)
            
            # Atualiza o estado
            self.previous_landmarks = landmarks
            self.frame_count += 1
            
            # Verifica se houve movimento suficiente e piscadas
            if self.frame_count >= MIN_FRAMES_FOR_DETECTION:
                has_movement = movement > MOVEMENT_THRESHOLD
                has_blinks = self.blink_count > 0
                
                # Atualiza informações de debug
                self.debug_info.update({
                    'movement': movement,
                    'blink_count': self.blink_count,
                    'frame_count': self.frame_count,
                    'has_movement': has_movement,
                    'has_blinks': has_blinks
                })
                
                if not has_movement:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        self.debug_info['reason'] = 'Movimento insuficiente'
                        return False
                elif not has_blinks:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        self.debug_info['reason'] = 'Nenhuma piscada detectada'
                        return False
                else:
                    self.consecutive_failures = 0
                
                return True
                
            return True
            
        except Exception as e:
            print(f"Erro na detecção de vivacidade: {str(e)}")
            if self.frame_count < INITIALIZATION_FRAMES:
                return True
            self.debug_info['reason'] = f'Erro: {str(e)}'
            return False

# Instância global do detector
liveness_detector = LivenessDetector()

def detect_liveness(frame):
    return liveness_detector.detect_liveness(frame)
