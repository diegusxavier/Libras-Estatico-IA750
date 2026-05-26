import cv2
import mediapipe as mp
import time
import os
import pickle
import numpy as np
from collections import deque, Counter 

# Caminhos relativos
MP_MODEL_PATH = 'models/hand_landmarker.task'
ML_MODEL_PATH = 'models/modelo_rf_libras.pkl'

def load_ml_model():
    if not os.path.exists(ML_MODEL_PATH):
        print(f"[ERRO] Modelo de ML não encontrado em: {ML_MODEL_PATH}")
        print("Execute o script '03_treinamento.py' primeiro.")
        return None
    
    with open(ML_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("[INFO] Modelo de Machine Learning carregado com sucesso!")
    return model

def draw_landmarks(image, hand_landmarks):
    """Desenha os landmarks e conexões na mão."""
    h, w, _ = image.shape
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    points = []
    for landmark in hand_landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    for start_idx, end_idx in connections:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(image, points[start_idx], points[end_idx], (255, 0, 0), 2)


def main():
    clf = load_ml_model()
    if clf is None: return

    if not os.path.exists(MP_MODEL_PATH):
        print(f"[ERRO] Modelo do MediaPipe não encontrado: {MP_MODEL_PATH}")
        return

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    print("\n[INFO] Tradutor iniciado.")


    # Tamanho do buffer para suavizar as previsões e evitar oscilações
    BUFFER_SIZE = 15  # Analisa aprox. meio segundo de vídeo (15 frames)
    prediction_buffer = deque(maxlen=BUFFER_SIZE)
    
    palavra_formada = ""
    ultima_letra_adicionada = ""
    frames_sem_mao = 0 # Contador para saber quando o utilizador baixou a mão

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
            letra_atual_tela = "Aguardando sinal..."

            if result.hand_landmarks:
                frames_sem_mao = 0 # Resetamos o contador de ausência
                hand_landmarks = result.hand_landmarks[0]
                draw_landmarks(frame, hand_landmarks)

                # Extração normalizada
                base_x = hand_landmarks[0].x
                base_y = hand_landmarks[0].y
                base_z = hand_landmarks[0].z

                row = []
                for landmark in hand_landmarks:
                    rel_x = landmark.x - base_x
                    rel_y = landmark.y - base_y
                    rel_z = landmark.z - base_z
                    row.extend([rel_x, rel_y, rel_z])
                
                X_realtime = np.array([row])
                prediction = clf.predict(X_realtime)[0] # Pega a string da previsão
                
                # Exibe o que o modelo está a pensar neste exato milissegundo
                letra_atual_tela = f"Lendo: {prediction}"

                # ==========================================
                # LÓGICA DO BUFFER E MÁQUINA DE ESTADOS
                # ==========================================
                prediction_buffer.append(prediction)

                # Só toma uma decisão se o buffer estiver cheio
                if len(prediction_buffer) == BUFFER_SIZE:
                    # Conta qual letra apareceu mais vezes no buffer
                    contagem = Counter(prediction_buffer)
                    letra_mais_comum, qtd = contagem.most_common(1)[0]

                    # Confiança de 70%: A letra tem de aparecer em 11 dos 15 frames
                    if qtd >= (BUFFER_SIZE * 0.7):
                        # Só adiciona à palavra se for uma letra nova
                        if letra_mais_comum != ultima_letra_adicionada:
                            palavra_formada += letra_mais_comum
                            ultima_letra_adicionada = letra_mais_comum
                            
            else:
                # Se não há mãos detectadas
                letra_atual_tela = "Nenhuma mao detectada"
                frames_sem_mao += 1
                
                # Se a mão sumir por cerca de 1 segundo (30 frames)
                if frames_sem_mao > 30:
                    ultima_letra_adicionada = "" # Limpa a última letra
                    prediction_buffer.clear()    # Limpa a memória recente
                    
                    # Se você quiser usar a ausência da mão como "Espaço" entre palavras:
                    if len(palavra_formada) > 0 and palavra_formada[-1] != " ":
                        palavra_formada += " "

            # ==========================================
            # INTERFACE GRÁFICA ATUALIZADA
            # ==========================================
            # Barra superior (Letra em tempo real)
            cv2.rectangle(frame, (0, 0), (640, 40), (50, 50, 50), -1)
            cv2.putText(frame, letra_atual_tela, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Barra inferior (A Palavra sendo construída)
            cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
            cv2.putText(frame, f"Palavra: {palavra_formada}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.imshow('Tradutor de Libras - IA750', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'): # Pressione 'c' para limpar a palavra
                palavra_formada = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()