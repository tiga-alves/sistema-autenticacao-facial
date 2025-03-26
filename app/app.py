import sys
import os
import streamlit as st
import cv2
import numpy as np

# Adiciona o diretório raiz ao caminho de importação
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Agora você pode importar os módulos corretamente
from custom_face_detection import detect_faces
from face_identification import identify_face, load_faces
from liveness_detection import detect_liveness

st.title("Sistema de Autenticação Facial")

# Carrega as faces conhecidas do banco de dados
try:
    load_faces()
    st.success("Faces conhecidas carregadas com sucesso!")
except Exception as e:
    st.error(f"Erro ao carregar faces conhecidas: {str(e)}")
    st.stop()

camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    frame = detect_faces(frame)
    frame = identify_face(frame)

    if not detect_liveness(frame):
        st.write("Fraude suspeita detectada!")
        break
    
    cv2.imshow("Autenticação", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()