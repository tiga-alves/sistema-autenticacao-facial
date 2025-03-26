import cv2
import numpy as np

# Limiar configurável para detecção de vivacidade
LAPLACIAN_THRESHOLD = 50

def detect_liveness(frame):
    if frame is None or frame.size == 0:
        raise ValueError("Frame inválido fornecido")
        
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Adiciona verificações adicionais para detecção de vivacidade mais robusta
        if laplacian < LAPLACIAN_THRESHOLD:
            return False  # Possível imagem estática
            
        # Verificação adicional para qualidade da imagem
        if cv2.mean(gray)[0] < 20 or cv2.mean(gray)[0] > 235:
            return False  # Possível imagem de baixa qualidade ou superexposta
            
        return True  # Movimento detectado (provável face real)
        
    except Exception as e:
        print(f"Erro na detecção de vivacidade: {str(e)}")
        return False  # Falha segura: retorna False em caso de erro
