import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- CONFIG ---
MODEL_PATH = './training_results/best_model.pth'
# IMAGE_PATH = './test_imgs/test1-001.jpg'
# IMAGE_PATH = './test_imgs/test2-002.jpg'
# IMAGE_PATH = './test_imgs/test3-003.jpg'
IMAGE_PATH = './test_imgs/test4-004.jpg'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# --- LOAD SAVED DATA ---
checkpoint = torch.load(MODEL_PATH)
class_names = checkpoint['classes']

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
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    print(f"Prediction: {class_names[predicted_idx]} ({confidence.item()*100:.2f}%)")

if __name__ == "__main__":
    if os.path.exists(IMAGE_PATH):
        predict(IMAGE_PATH)
    else:
        print("Image path not found.")