from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import cv2
import numpy as np
from custom_face_detection import detect_faces
from face_identification import identify_face, load_faces
from liveness_detection import detect_liveness


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicialização
    try:
        load_faces()
    except Exception as e:
        print(f"Erro ao carregar faces: {str(e)}")
    yield
    # Finalização
    pass

app = FastAPI(lifespan=lifespan)

@app.post("/authenticate/")
async def authenticate(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="O arquivo deve ser uma imagem")
    
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Arquivo vazio")
            
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Formato de imagem inválido")
            
        # Armazena o frame original para detecção de vivacidade
        original_frame = frame.copy()
        
        # Processa o frame para detecção e identificação facial
        frame = detect_faces(frame)
        frame = identify_face(frame)
        
        # Usa o frame original para detecção de vivacidade
        if not detect_liveness(original_frame):
            return {"status": "Fraude suspeita", "liveness": False}
        
        return {"status": "Autenticado com sucesso", "liveness": True}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Limpa os recursos
        if 'frame' in locals():
            del frame
        if 'original_frame' in locals():
            del original_frame
