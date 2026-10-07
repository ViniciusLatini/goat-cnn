import cv2
from ultralytics import YOLO

# 1. Configurações Iniciais
model = YOLO('yolov8m.pt')
video_path = "./testes/teste1-001.avi"
output_path = "sheep-test-1.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Pegar propriedades do vídeo original para o output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definir o codec e criar o objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("Processando... Pressione Ctrl+C para interromper e salvar o que foi feito.")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Predição focada na classe 18 (sheep)
        results = model(frame, classes=[18], verbose=False)
        
        # Desenha as detecções no frame
        annotated_frame = results[0].plot()

        # Escreve o frame no arquivo de saída
        out.write(annotated_frame)

        # Opcional: Mostrar progresso visual
        cv2.imshow("YOLOv8 Real-time", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupção detectada! Finalizando arquivo de vídeo...")

finally:
    # Este bloco sempre executa, garantindo que o arquivo não corrompa
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Vídeo salvo com sucesso em: {output_path}")