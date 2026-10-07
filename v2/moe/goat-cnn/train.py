import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import os
import time

# --- 1. SETTINGS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_EPOCHS = 50
DATA_DIR = './dataset'
OUTPUT_DIR = './training_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# --- 2. DATA LOADERS ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# --- 3. MODEL SETUP ---
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 4) 
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
early_stopping = EarlyStopping(patience=7)

# --- 4. TRAINING FUNCTION ---
def train_model(model, criterion, optimizer, num_epochs=50):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    final_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        epoch_start = time.time()
        final_epoch = epoch + 1
        print(f'Epoch {final_epoch}/{num_epochs}')
        
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                early_stopping(epoch_loss)
        
        epoch_end = time.time() - epoch_start
        print(f'Epoch duration: {epoch_end:.2f}s')

        if early_stopping.early_stop:
            print(f"--- Early stopping triggered at epoch {final_epoch} ---")
            break

    total_time = time.time() - since
    print(f'Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s')
    model.load_state_dict(best_model_wts)
    return model, history, final_epoch

# --- 5. RUN TRAINING ---
model, history, stop_epoch = train_model(model, criterion, optimizer, num_epochs=MAX_EPOCHS)

# --- 6. FINAL EVALUATION (Scope Fix) ---
model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for inputs, labels in dataloaders['val']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# --- 7. SAVE RESULTS ---
# Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.title('Loss per Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Val')
plt.title('Accuracy per Epoch')
plt.legend()
plt.savefig(f'{OUTPUT_DIR}/learning_curves.png')

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png')

# Metrics Report
report_text = classification_report(y_true, y_pred, target_names=class_names)
with open(f'{OUTPUT_DIR}/metrics_report.txt', 'w') as f:
    f.write(f"TRAINING SUMMARY\n")
    f.write(f"================\n")
    f.write(f"Status: {'Early Stopped' if early_stopping.early_stop else 'Completed Max Epochs'}\n")
    f.write(f"Stop Epoch: {stop_epoch}\n")
    f.write(f"Best Val Acc: {max(history['val_acc']):.4f}\n\n")
    f.write(f"CLASSIFICATION REPORT:\n")
    f.write(report_text)

# Model Weights
torch.save({'model_state_dict': model.state_dict(), 'classes': class_names}, f'{OUTPUT_DIR}/best_model.pth')
print(f"All files saved in {OUTPUT_DIR}")