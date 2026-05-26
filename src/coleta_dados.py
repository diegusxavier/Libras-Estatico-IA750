import cv2
import mediapipe as mp
import time
import os
import csv

# Caminhos relativos considerando execução a partir de src/
MODEL_PATH = 'models/hand_landmarker.task'
DATASET_PATH = 'data/dataset_libras.csv'

def draw_landmarks(image, hand_landmarks):
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


def save_landmarks_to_csv(label, hand_landmarks):
    """
    Salva as coordenadas no formato RELATIVO.
    O pulso (landmark 0) vira a âncora (0, 0, 0) e todos os outros
    pontos são salvos como a distância em relação ao pulso.
    """
    file_exists = os.path.isfile(DATASET_PATH)
    row = [label]
    
    # 1. Define a âncora (pulso é o primeiro ponto da lista, índice 0)
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    base_z = hand_landmarks[0].z
    
    # 2. Calcula e salva a posição relativa de todos os 21 pontos
    for landmark in hand_landmarks:
        rel_x = landmark.x - base_x
        rel_y = landmark.y - base_y
        rel_z = landmark.z - base_z
        row.extend([rel_x, rel_y, rel_z])
        
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    
    with open(DATASET_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ['label']
            for i in range(21):
                header.extend([f'x{i}', f'y{i}', f'z{i}'])
            writer.writerow(header)
            
        writer.writerow(row)



def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERRO] Modelo não encontrado em: {MODEL_PATH}")
        return

    # Interação com o usuário para definir os parâmetros da coleta
    print("=== Módulo de Coleta de Dados ===")
    label = input("Qual letra/sinal você vai gravar agora? (Ex: A): ").strip().upper()
    # try:
    #     num_samples = int(input("Quantas amostras (frames) deseja gravar? (Recomendado: 300 a 500): "))
    # except ValueError:
    #     print("[ERRO] Número de amostras inválido. Usando 300 como padrão.")
    num_samples = 300

    # Configuração da Tasks API (exatamente como no script base)
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1, # Para dactilologia básica, focaremos em 1 mão por vez
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Webcam não acessível.")
        return


    print(f"\n[INFO] Posicione sua mão para o sinal '{label}'.")
    print("Pressione 's' para INICIAR a gravação ou 'q' para CANCELAR.")

    samples_collected = 0
    recording = False

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # Só processa se encontrar exatamente uma mão
            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0] # Pega a primeira mão detectada
                draw_landmarks(frame, hand_landmarks)

                if recording:
                    # Salva os dados no CSV
                    save_landmarks_to_csv(label, hand_landmarks)
                    samples_collected += 1
                    
                    # Exibe o progresso na tela do OpenCV
                    cv2.putText(frame, f"Gravando '{label}': {samples_collected}/{num_samples}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Avisos na tela quando não estiver gravando
            if not recording:
                cv2.putText(frame, f"Sinal: {label} - Pressione 's' para gravar", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Coleta de Dados - MVP Libras', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Coleta cancelada pelo usuário.")
                break
            elif key == ord('s') and not recording:
                print("\n[INFO] GRAVAÇÃO INICIADA!")
                recording = True

            # Para automaticamente ao atingir o limite
            if samples_collected >= num_samples:
                print(f"\n[SUCESSO] {num_samples} amostras da classe '{label}' coletadas!")
                break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()