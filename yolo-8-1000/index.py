from pathlib import Path
from ultralytics import YOLO
import torch

YAML_FILENAME = "data_custom.yaml"
NUM_CLASSES = 1
CLASS_NAMES = ["goat"]

MODEL_TO_TRAIN = 'yolov8m.pt'
EPOCHS = 300
IMAGE_SIZE = 640
BATCH_SIZE = 8

PREDICTION_TEST_FOLDER = "./testes/teste4-005.avi"
CONFIDENCE_THRESHOLD = 0.8 #0.9

def create_yaml_file(train_path, val_path):
    print("\n Criando arquivo de configuração .yaml...")

    yaml_content = f"""
train: {Path(train_path) / 'images'}
val: {Path(val_path) / 'images'}

nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""

    with open(YAML_FILENAME, 'w') as f:
        f.write(yaml_content)

    print(f"Arquivo '{YAML_FILENAME}' criado com sucesso.")


def train_model():
    print("\n Iniciando o treinamento do modelo YOLO...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo detectado: {device.upper()}")
    if device == 'cpu':
        print("Nenhuma GPU detectada")

    model = YOLO(MODEL_TO_TRAIN)


    results = model.train(
        data=YAML_FILENAME,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=30,
        device=device,
        workers=8
    )

    print("Treinamento concluído!")
    print(f"Resultados e o melhor modelo foram salvos na pasta '{results.save_dir}'.")
    return results.save_dir / 'weights' / 'best.pt'


def run_prediction(best_model_path):
    if not PREDICTION_TEST_FOLDER:
        print("\nNenhuma pasta de predição foi definida. Encerrando a predição.")
        return

    print(f"\n Executando predição na pasta: '{PREDICTION_TEST_FOLDER}'...")

    if not best_model_path.exists():
        print(f"Erro: Modelo treinado '{best_model_path}' não encontrado.")
        return

    model = YOLO(best_model_path)

    print("\n   --- Predição com Rótulos e Confiança ---")
    model.predict(
        source=PREDICTION_TEST_FOLDER,
        show=True,
        conf=CONFIDENCE_THRESHOLD,
        save=True,
        save_txt=True
    )

    print("\n   --- Predição OCULTANDO Rótulos e Confiança ---")
    model.predict(
        source=PREDICTION_TEST_FOLDER,
        show=True,
        conf=CONFIDENCE_THRESHOLD,
        save=True,
        save_txt=True,
        hide_labels=True,
        hide_conf=True
    )

    print("\n Predição concluída. Os resultados foram salvos na pasta 'runs/detect/'.")

if __name__ == '__main__':

    train_path = "./data/train"
    val_path = "./data/val"

    # create_yaml_file(train_path, val_path);

    # best_model_path = train_model()
    
    best_model_path = Path("./runs/detect/k-fold/fold_5/weights/best.pt")
    
    run_prediction(best_model_path)