import cv2
import mediapipe as mp
import time
import os

# Caminho relativo considerando que o script roda de dentro da pasta src/
MODEL_PATH = 'models\hand_landmarker.task'

def draw_landmarks(image, hand_landmarks):
    """
    Mapeia os landmarks da mão e desenha os pontos e conexões 
    diretamente com OpenCV.
    """
    h, w, _ = image.shape
    
    # Grafo de conexões anatômicas da mão (índices de 0 a 20)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Polegar
        (0, 5), (5, 6), (6, 7), (7, 8),        # Indicador
        (5, 9), (9, 10), (10, 11), (11, 12),   # Dedo médio
        (9, 13), (13, 14), (14, 15), (15, 16), # Anelar
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Mínimo e palma
    ]

    # Converte coordenadas normalizadas (0.0 - 1.0) para pixels absolutos
    points = []
    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1) # Ponto verde

    # Desenha as linhas entre os pontos conectados
    for start_idx, end_idx in connections:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(image, points[start_idx], points[end_idx], (255, 0, 0), 2) # Linha azul

def main():
    # Prevenção de falha: verifica se o modelo está no lugar certo
    if not os.path.exists(MODEL_PATH):
        print(f"[ERRO] Modelo não encontrado em: {MODEL_PATH}")
        return

    # Aliases para encurtar as chamadas da Tasks API
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Configuração do detector
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO, # Otimizado para fluxo contínuo
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Webcam não acessível.")
        return

    print("[INFO] Captura iniciada. Pressione 'q' para fechar.")



# Usamos 'with' para garantir que os recursos do modelo sejam liberados no fim
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Espelhamento do frame (efeito de espelho)
            frame = cv2.flip(frame, 1)

            # Conversão BGR (padrão do OpenCV) para RGB (exigência do MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encapsula na estrutura de Imagem do MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # O modo VIDEO exige um timestamp exato em milissegundos
            frame_timestamp_ms = int(time.time() * 1000)

            # Executa a inferência
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # Desenha caso encontre mãos
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)

            # Exibe a janela
            cv2.imshow('MVP Libras - IA750', frame)

            # Condição de saída
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Limpeza dos recursos do OpenCV
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()