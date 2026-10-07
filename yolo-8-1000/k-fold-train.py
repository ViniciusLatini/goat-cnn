from pathlib import Path
from ultralytics import YOLO
import torch
from sklearn.model_selection import KFold
import shutil


YAML_FILENAME_TEMPLATE = "data_fold_{}.yaml"
NUM_CLASSES = 1
CLASS_NAMES = ["goat"]

MODEL_TO_TRAIN = 'yolov8m.pt'
EPOCHS = 300
IMAGE_SIZE = 640
BATCH_SIZE = -1 # auto batch size
K_FOLDS = 5  # splits

PREDICTION_TEST_FOLDER = "./testes"
CONFIDENCE_THRESHOLD = 0.8

def get_all_image_paths(train_dir, val_dir):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    
    for directory in [train_dir, val_dir]:
        path_obj = Path(directory)
        if path_obj.exists():
            search_path = path_obj / 'images' if (path_obj / 'images').exists() else path_obj
            for ext in exts:
                images.extend(list(search_path.rglob(ext)))
    
    return sorted(list(set(images)))

def create_fold_configurations(fold_index, train_indices, val_indices, all_images):
    train_files = [str(all_images[i].absolute()) for i in train_indices]
    val_files = [str(all_images[i].absolute()) for i in val_indices]
    
    txt_train_name = f"train_fold_{fold_index}.txt"
    txt_val_name = f"val_fold_{fold_index}.txt"
    
    with open(txt_train_name, 'w') as f:
        f.write('\n'.join(train_files))
        
    with open(txt_val_name, 'w') as f:
        f.write('\n'.join(val_files))
    
    yaml_content = f"""
train: {Path(txt_train_name).absolute()}
val: {Path(txt_val_name).absolute()}

nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""
    current_yaml = YAML_FILENAME_TEMPLATE.format(fold_index)
    with open(current_yaml, 'w') as f:
        f.write(yaml_content)
        
    return current_yaml

def train_cross_validation(train_path, val_path):
    print(f"\n Iniciando Cross-Validation com K={K_FOLDS}...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo detectado: {device.upper()}")

    all_images = get_all_image_paths(train_path, val_path)
    print(f"Total de imagens encontradas: {len(all_images)}")
    
    if len(all_images) == 0:
        print("Erro: Nenhuma imagem encontrada nos diretórios.")
        return None

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    results_metrics = []
    best_map = 0
    best_model_path_overall = None

    for i, (train_index, val_index) in enumerate(kf.split(all_images)):
        print(f"\n--- Iniciando Fold {i+1}/{K_FOLDS} ---")
        
        yaml_file = create_fold_configurations(i, train_index, val_index, all_images)
        
        model = YOLO(MODEL_TO_TRAIN)
        
        project_name = "runs/detect"
        run_name = f"fold_{i+1}"

        results = model.train(
            data=yaml_file,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            patience=30,
            device=device,
            workers=8,
            project=project_name,
            name=run_name,
            exist_ok=True
        )
        
        metrics = model.val()
        map50 = metrics.box.map50
        print(f"Fold {i+1} mAP50: {map50:.4f}")
        results_metrics.append(map50)

        if map50 > best_map:
            best_map = map50
            best_model_path_overall = Path(results.save_dir) / 'weights' / 'best.pt'

    print("\n--- Resumo da Cross-Validation ---")
    avg_map = sum(results_metrics) / len(results_metrics)
    print(f"Média mAP50: {avg_map:.4f}")
    print(f"Melhor modelo global: {best_model_path_overall}")
    
    return best_model_path_overall

def run_prediction(best_model_path):
    if not PREDICTION_TEST_FOLDER:
        return

    if not best_model_path or not best_model_path.exists():
        print(f"Erro: Modelo '{best_model_path}' não encontrado.")
        return

    print(f"\n Executando predição usando o melhor modelo do Cross Validation...")
    print(f"Modelo: {best_model_path}")

    model = YOLO(best_model_path)

    model.predict(
        source=PREDICTION_TEST_FOLDER,
        show=True,
        conf=CONFIDENCE_THRESHOLD,
        save=True,
        save_txt=True,
        project="runs/detect",
        name="prediction_results",
        exist_ok=True
    )
    
    model.predict(
        source=PREDICTION_TEST_FOLDER,
        show=True,
        conf=CONFIDENCE_THRESHOLD,
        save=True,
        save_txt=True,
        hide_labels=True,
        hide_conf=True,
        project="runs/detect",
        name="prediction_results_clean",
        exist_ok=True
    )

    print("\n Predição concluída.")

if __name__ == '__main__':
    train_path = "./data/train"
    val_path = "./data/val"
    # best_model = train_cross_validation(train_path, val_path)
    best_model = Path("runs/detect/train3/weights/best.pt")

    if best_model:
        run_prediction(best_model)