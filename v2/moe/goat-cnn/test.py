import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- CONFIG ---
MODEL_PATH = './training_results/best_model.pth'
IMAGE_PATH = './test_imgs/teste-6.jpeg'

# Explicitly set device to CPU
device = torch.device("cpu")

# --- LOAD SAVED DATA ---
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint['classes']
else:
    print(f"Model not found at {MODEL_PATH}")
    exit()

# --- REBUILD MODEL ---
model = models.efficientnet_b0()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 4)


model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# --- IMAGE PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_t)
            # Apply softmax to get percentages
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            print(f"Probabilities: {probabilities}")
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        print(f"\n--- Inference Result ---")
        print(f"File: {os.path.basename(img_path)}")
        print(f"Prediction: {class_names[predicted_idx]}")
        print(f"Confidence: {confidence.item()*100:.2f}%")
        print(f"------------------------\n")
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    if os.path.exists(IMAGE_PATH):
        predict(IMAGE_PATH)
    else:
        print(f"Image path not found: {IMAGE_PATH}")