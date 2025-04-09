import sys
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Adiciona o diretório raiz ao caminho de importação
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa os módulos
from custom_face_detection import detect_faces
from face_identification import identify_face, load_faces
from liveness_detection import detect_liveness

def main():
    st.title("Sistema de Autenticação Facial")
    
    # Inicializa o estado da sessão
    if 'camera' not in st.session_state:
        st.session_state.camera = None
    if 'fraude_detectada' not in st.session_state:
        st.session_state.fraude_detectada = False
    if 'last_frame_time' not in st.session_state:
        st.session_state.last_frame_time = 0
    if 'frame_interval' not in st.session_state:
        st.session_state.frame_interval = 0.05  # 50ms entre frames
    if 'consecutive_failures' not in st.session_state:
        st.session_state.consecutive_failures = 0
    if 'max_consecutive_failures' not in st.session_state:
        st.session_state.max_consecutive_failures = 3
    
    # Carrega as faces conhecidas do banco de dados
    try:
        load_faces()
        st.success("Faces conhecidas carregadas com sucesso!")
    except Exception as e:
        st.error(f"Erro ao carregar faces conhecidas: {str(e)}")
        st.stop()
    
    # Interface do usuário
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Câmera")
        camera_placeholder = st.empty()
    
    with col2:
        st.subheader("Status")
        status_placeholder = st.empty()
        if st.button("Sair", key="exit_button_main"):
            st.session_state.camera = None
            st.experimental_rerun()
    
    # Inicializa a câmera se ainda não estiver inicializada
    if st.session_state.camera is None:
        st.session_state.camera = cv2.VideoCapture(0)
        if not st.session_state.camera.isOpened():
            st.error("Erro ao abrir a câmera. Por favor, verifique se ela está conectada.")
            st.stop()
        
        # Configura a câmera para melhor performance
        st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        st.session_state.camera.set(cv2.CAP_PROP_FPS, 30)
        st.session_state.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimiza o buffer
    
    try:
        while True:
            current_time = time.time()
            if current_time - st.session_state.last_frame_time < st.session_state.frame_interval:
                time.sleep(0.001)  # Pausa mínima para não sobrecarregar a CPU
                continue
                
            ret, frame = st.session_state.camera.read()
            if not ret:
                st.error("Erro ao ler o frame da câmera")
                break
            
            # Processa o frame
            frame = detect_faces(frame)
            frame = identify_face(frame)
            
            # Verifica vivacidade
            if not detect_liveness(frame):
                st.session_state.consecutive_failures += 1
                if st.session_state.consecutive_failures >= st.session_state.max_consecutive_failures:
                    st.session_state.fraude_detectada = True
                    status_placeholder.error("⚠️ Fraude suspeita detectada!")
                    break
                status_placeholder.warning("⚠️ Movimento suspeito detectado...")
            else:
                st.session_state.consecutive_failures = 0
                status_placeholder.success("✅ Sistema funcionando normalmente")
            
            # Converte o frame para exibição
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Atualiza a interface
            camera_placeholder.image(frame_pil, use_column_width=True)
            
            # Atualiza o tempo do último frame
            st.session_state.last_frame_time = current_time
            
    except Exception as e:
        st.error(f"Erro durante a execução: {str(e)}")
    finally:
        # Limpa os recursos
        if st.session_state.camera is not None:
            st.session_state.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()