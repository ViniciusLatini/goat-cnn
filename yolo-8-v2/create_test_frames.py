import cv2
import os

# Configurações
SOURCE_FOLDER = './testes'
DEST_FOLDER = './primeiros_frames'

# Cria a pasta de destino se não existir
os.makedirs(DEST_FOLDER, exist_ok=True)

print(f"Iniciando extração em: {SOURCE_FOLDER}")

# Percorre os arquivos da pasta
for file_name in os.listdir(SOURCE_FOLDER):
    if file_name.lower().endswith('.avi'):
        video_path = os.path.join(SOURCE_FOLDER, file_name)
        
        # Abre o vídeo
        cap = cv2.VideoCapture(video_path)
        
        # Tenta ler o primeiro frame
        ret, frame = cap.read()
        
        if ret:
            # Define o nome da imagem (ex: video1.avi -> video1.jpg)
            img_name = os.path.splitext(file_name)[0] + ".jpg"
            save_path = os.path.join(DEST_FOLDER, img_name)
            
            # Salva o frame
            cv2.imwrite(save_path, frame)
            print(f"Sucesso: {img_name} extraído.")
        else:
            print(f"Erro: Não foi possível ler o frame de {file_name}")
        
        cap.release()

print("\nProcesso finalizado!")