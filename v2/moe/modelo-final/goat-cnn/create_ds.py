import cv2
import os
import random
from sklearn.model_selection import train_test_split

BASE_PATH = './videos'
OUTPUT_PATH = './dataset'
TARGET_IMAGES = 1000  
TRAIN_RATIO = 0.8
EXTENSIONS = ('.avi', '.mp4')

def get_video_info(class_folder):
    video_files = [f for f in os.listdir(class_folder) if f.lower().endswith(EXTENSIONS)]
    total_frames = 0
    for v in video_files:
        cap = cv2.VideoCapture(os.path.join(class_folder, v))
        total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return video_files, total_frames

classes = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]

for cls in classes:
    cls_folder = os.path.join(BASE_PATH, cls)
    video_files, total_available = get_video_info(cls_folder)
    
    if total_available == 0: continue

    step = max(1, total_available // TARGET_IMAGES)
    class_frames = []
    
    for v_file in video_files:
        cap = cv2.VideoCapture(os.path.join(cls_folder, v_file))
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or len(class_frames) >= TARGET_IMAGES: break
            if count % step == 0:
                class_frames.append(frame)
            count += 1
        cap.release()

  
    train_f, val_f = train_test_split(class_frames, test_size=0.2, random_state=42)
    
    for split, frames in [('train', train_f), ('val', val_f)]:
        split_dir = os.path.join(OUTPUT_PATH, split, cls)
        os.makedirs(split_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(split_dir, f"{cls}_{i}.jpg"), frame)

print("\n" + "="*30)
print("RESUMO DO DATASET FINAL")
print("="*30)

total_train = 0
total_val = 0

for split in ['train', 'val']:
    print(f"\nConjunto de {split.upper()}:")
    for cls in classes:
        path = os.path.join(OUTPUT_PATH, split, cls)
        num_files = len(os.listdir(path)) if os.path.exists(path) else 0
        print(f"  - {cls}: {num_files} imagens")
        if split == 'train': total_train += num_files
        else: total_val += num_files

print("-" * 30)
print(f"TOTAL TREINO: {total_train}")
print(f"TOTAL VALIDAÇÃO: {total_val}")
print(f"TOTAL GERAL: {total_train + total_val}")
print("="*30)
