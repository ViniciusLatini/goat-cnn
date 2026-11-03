import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import models, transforms
from captum.attr import Occlusion

# --- CONFIG ---
MODEL_PATH = './training_results/best_model.pth'
INPUT_IMAGES_DIR = './test_imgs'
OUTPUT_DIR = './occlusion_results'
CLASS_NAMES = ['C1: Frente Dia', 'C2: Frente Noite', 'C3: Cima Dia', 'C4: Cima Noite']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation matching the training verification set
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_my_model(path):
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 4)

    checkpoint = torch.load(path, map_location=DEVICE)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Pesos da EfficientNet carregados com sucesso!")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Pesos carregados (formato direto).")

    model.to(DEVICE)
    model.eval()
    return model

def run_occlusion_study():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model = load_my_model(MODEL_PATH)
    occlusion = Occlusion(model)

    # Class mappings with filenames and corresponding labels for paper reference
    CLASS_MAPPING = {
        0: {"name": "C1: Frente Dia", "file": "test1-001.jpg", "label": "(a) C1: Frente Dia"},
        1: {"name": "C2: Frente Noite", "file": "test3-003.jpg", "label": "(b) C2: Frente Noite"},
        2: {"name": "C3: Cima Dia", "file": "test2-002.jpg", "label": "(c) C3: Cima Dia"},
        3: {"name": "C4: Cima Noite", "file": "test4-004.jpg", "label": "(d) C4: Cima Noite"}
    }

    results = []

    for idx in sorted(CLASS_MAPPING.keys()):
        info = CLASS_MAPPING[idx]
        img_path = os.path.join(INPUT_IMAGES_DIR, info["file"])
        if not os.path.exists(img_path):
            print(f"⚠️ Imagem não encontrada: {img_path}")
            continue

        raw_image = Image.open(img_path).convert('RGB')
        input_tensor = transform(raw_image).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            pred_label_idx = torch.max(output, 1)[1].item()

        # Compute Occlusion Map (Attribution)
        print(f"Processando atribuição do Occlusion para {info['file']} (Classe {info['name']})...")
        attributions = occlusion.attribute(input_tensor,
                                           strides=(3, 8, 8),
                                           target=pred_label_idx,
                                           sliding_window_shapes=(3, 15, 15),
                                           baselines=0)

        attributions_np = attributions.squeeze().cpu().detach().numpy()
        attributions_np = np.transpose(attributions_np, (1, 2, 0))
        heatmap = np.sum(np.abs(attributions_np), axis=2)

        # Unnormalize original image for display
        img_show = input_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        img_show = (img_show * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_show = np.clip(img_show, 0, 1)

        results.append({
            "class_idx": idx,
            "name": info["name"],
            "label": info["label"],
            "file": info["file"],
            "pred_idx": pred_label_idx,
            "img": img_show,
            "heatmap": heatmap
        })

        # Save individual result (original + heatmap)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_show)
        plt.title(f"Original ({info['file']}) - Pred: {CLASS_NAMES[pred_label_idx]}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(heatmap, cmap='hot')
        plt.title("Mapa de Oclusão")
        plt.colorbar()
        plt.axis('off')

        save_name = f"occlusion_{info['file']}"
        plt.savefig(os.path.join(OUTPUT_DIR, save_name), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✓ Salvo individual: {save_name}")

    if not results:
        print("Nenhum resultado processado.")
        return

    # --- 1. GENERATE COMBINED BLENDED GRID (2x2) ---
    print("Gerando imagem combinada com heatmaps sobrepostos (2x2)...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    for i, res in enumerate(results):
        ax = axes[i]
        ax.imshow(res["img"])
        
        # Normalize heatmap to [0, 1] for best display
        h = res["heatmap"]
        h_min, h_max = h.min(), h.max()
        h_norm = (h - h_min) / (h_max - h_min) if h_max > h_min else h
        
        # Overlay the heatmap
        ax.imshow(h_norm, cmap='jet', alpha=0.5)
        
        ax.set_title(f"Predição: {CLASS_NAMES[res['pred_idx']]}", fontsize=12, pad=10)
        ax.text(0.5, -0.1, res["label"], transform=ax.transAxes, 
                ha="center", va="center", fontsize=12, fontweight="bold")
        ax.axis('off')

    plt.tight_layout()
    blended_output_path = os.path.join(OUTPUT_DIR, 'occlusion_2x2_blended.png')
    plt.savefig(blended_output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Salvo grid sobreposto: {blended_output_path}")

    # --- 2. GENERATE COMBINED SIDE-BY-SIDE GRID (2x2 of Original|Heatmap) ---
    print("Gerando imagem combinada lado a lado (2x2)...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, res in enumerate(results):
        ax = axes[i]
        
        # Normalize heatmap
        h = res["heatmap"]
        h_min, h_max = h.min(), h.max()
        h_norm = (h - h_min) / (h_max - h_min) if h_max > h_min else h
        
        # Convert heatmap to RGB using jet colormap
        heatmap_rgb = cm.jet(h_norm)[:, :, :3]
        
        # Create a horizontal separator (white line)
        separator = np.ones((224, 4, 3))
        
        # Stack horizontally: Original | Separator | Heatmap
        combined_img = np.hstack((res["img"], separator, heatmap_rgb))
        
        ax.imshow(combined_img)
        
        ax.text(0.5, -0.1, res["label"], transform=ax.transAxes, 
                ha="center", va="center", fontsize=12, fontweight="bold")
        ax.axis('off')

    plt.tight_layout()
    side_by_side_output_path = os.path.join(OUTPUT_DIR, 'occlusion_2x2_side_by_side.png')
    plt.savefig(side_by_side_output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Salvo grid lado a lado: {side_by_side_output_path}")

if __name__ == "__main__":
    run_occlusion_study()
