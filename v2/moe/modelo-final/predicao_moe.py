#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline de Predição MoE (Mixture of Experts) - CNN + YOLOv8
Este script carrega um classificador CNN (EfficientNet-B0) para identificar o cenário da imagem/vídeo 
e executa o modelo especialista YOLO correspondente para fazer a predição.
"""

import os
import sys
import argparse
import time
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

# Mapeamento dos índices/classes da CNN para os pesos especialistas do YOLO
# class_1 -> Frente Dia
# class_2 -> Frente Noite
# class_3 -> Cima Dia
# class_4 -> Cima Noite
YOLO_WEIGHTS_MAP = {
    0: {"name": "best-front-dia.pt", "label": "Frente Dia"},
    1: {"name": "best-front-noite.pt", "label": "Frente Noite"},
    2: {"name": "best-sup-dia.pt", "label": "Cima Dia"},
    3: {"name": "best-sup-noite.pt", "label": "Cima Noite"}
}

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Predição MoE: Classificação por CNN + Detecção por YOLO especialista.")
    parser.add_argument("--input", "-i", required=True, type=str, help="Caminho para o arquivo de entrada (imagem ou vídeo)")
    parser.add_argument("--output", "-o", default="resultados", type=str, help="Caminho para o diretório de saída")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        print(f"Erro: Arquivo de entrada não encontrado: {input_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Definir dispositivo (CUDA se disponível, caso contrário CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo selecionado: {device}")

    # --- 1. CARREGAR MODELO CNN ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cnn_model_path = os.path.join(script_dir, "cnn-weight", "best_model.pth")

    if not os.path.exists(cnn_model_path):
        print(f"Erro: Pesos da CNN não encontrados em: {cnn_model_path}")
        sys.exit(1)

    print("Carregando modelo de classificação CNN...")
    try:
        checkpoint = torch.load(cnn_model_path, map_location=device)
        class_names = checkpoint.get('classes', ['class_1', 'class_2', 'class_3', 'class_4'])
        
        # Reconstrói a EfficientNet-B0
        cnn_model = models.efficientnet_b0(weights=None)
        num_ftrs = cnn_model.classifier[1].in_features
        cnn_model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
        
        # Carrega os pesos
        if 'model_state_dict' in checkpoint:
            cnn_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            cnn_model.load_state_dict(checkpoint)
            
        cnn_model = cnn_model.to(device)
        cnn_model.eval()
        print("✓ Classificador CNN carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar a CNN: {e}")
        sys.exit(1)

    # --- 2. DETERMINAR TIPO DE ENTRADA E EXTRAIR FRAME ---
    # Extensões comuns
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    
    _, ext = os.path.splitext(input_path.lower())
    is_image = ext in image_extensions
    is_video = ext in video_extensions

    if not is_image and not is_video:
        # Tenta inferir abrindo com o OpenCV
        cap = cv2.VideoCapture(input_path)
        if cap.isOpened():
            is_video = True
            cap.release()
        else:
            is_image = True

    # Prepara o transform da imagem para a EfficientNet
    cnn_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    frame_to_classify = None
    input_type_str = ""

    if is_image:
        input_type_str = "Imagem"
        print(f"Entrada identificada como Imagem: {os.path.basename(input_path)}")
        try:
            frame_to_classify = Image.open(input_path).convert('RGB')
        except Exception as e:
            print(f"Erro ao abrir imagem: {e}")
            sys.exit(1)
    else:
        input_type_str = "Vídeo"
        print(f"Entrada identificada como Vídeo: {os.path.basename(input_path)}")
        try:
            cap = cv2.VideoCapture(input_path)
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                print("Erro: Não foi possível ler o primeiro frame do vídeo.")
                sys.exit(1)
            cap.release()
            
            # Converte de BGR para RGB para o PyTorch/PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_to_classify = Image.fromarray(frame_rgb)
            print("✓ Primeiro frame extraído com sucesso para classificação.")
        except Exception as e:
            print(f"Erro ao ler frame do vídeo: {e}")
            sys.exit(1)

    # --- 3. CLASSIFICAR COM A CNN ---
    try:
        input_tensor = cnn_transform(frame_to_classify).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = cnn_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx_tensor = torch.max(probabilities, 0)
            predicted_idx = predicted_idx_tensor.item()
            
        cnn_pred_class = class_names[predicted_idx]
        info_especialista = YOLO_WEIGHTS_MAP.get(predicted_idx, {"name": "unknown.pt", "label": "Desconhecido"})
        
        print("\n" + "="*40)
        print("RESULTADO DA CLASSIFICAÇÃO CNN")
        print("="*40)
        print(f"Cenário Predito: {info_especialista['label']} ({cnn_pred_class})")
        print(f"Confiança:       {confidence.item()*100:.2f}%")
        print("="*40 + "\n")
    except Exception as e:
        print(f"Erro durante a predição da CNN: {e}")
        sys.exit(1)

    # --- 4. CARREGAR MODELO ESPECIALISTA YOLO ---
    yolo_filename = info_especialista["name"]
    yolo_weights_dir = os.path.join(script_dir, "best-weights")
    yolo_model_path = os.path.join(yolo_weights_dir, yolo_filename)

    if not os.path.exists(yolo_model_path):
        print(f"Erro: Pesos do YOLO especialista não encontrados em: {yolo_model_path}")
        sys.exit(1)

    print(f"Carregando modelo YOLO especialista: {yolo_filename}...")
    try:
        yolo_model = YOLO(yolo_model_path)
        print("✓ Modelo YOLO carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo YOLO: {e}")
        sys.exit(1)

    # --- 5. RODAR PREDIÇÃO NO YOLO ---
    # Vamos salvar os resultados diretamente na subpasta do output
    output_name = "predicao_moe"
    print(f"Iniciando predição no YOLO para {input_type_str} de entrada...")
    
    start_time = time.time()
    try:
        # Configurar execução do YOLO.
        # save=True salva o vídeo/imagem anotada
        # project=output_dir e name=output_name salvam o resultado em output_dir/output_name
        # exist_ok=True sobrescreve ou complementa a pasta sem criar sufixos incrementais (ex: predicao_moe2)
        yolo_device = device.index if device.type == 'cuda' else 'cpu'
        
        results = yolo_model.predict(
            source=input_path,
            save=True,
            project=output_dir,
            name=output_name,
            exist_ok=True,
            device=yolo_device,
            conf=0.7
        )
        elapsed_time = time.time() - start_time
        print(f"\n✓ Predição do YOLO concluída em {elapsed_time:.2f}s!")
    except Exception as e:
        print(f"Erro durante a execução do YOLO: {e}")
        sys.exit(1)

    # --- 6. GERAR RELATÓRIO / METADADOS ---
    save_dir = os.path.join(output_dir, output_name)
    report_path = os.path.join(save_dir, "resumo.txt")
    
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=========================================\n")
            f.write("RELATÓRIO DE EXECUÇÃO - PIPELINE MOE\n")
            f.write("=========================================\n")
            f.write(f"Data/Hora:         {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Arquivo Entrada:   {os.path.basename(input_path)}\n")
            f.write(f"Caminho Entrada:   {input_path}\n")
            f.write(f"Tipo de Entrada:   {input_type_str}\n")
            f.write(f"Diretório Saída:   {save_dir}\n")
            f.write("-----------------------------------------\n")
            f.write("CLASSIFICAÇÃO CNN (Cenário):\n")
            f.write(f"  - Classe:        {cnn_pred_class}\n")
            f.write(f"  - Descrição:     {info_especialista['label']}\n")
            f.write(f"  - Confiança:     {confidence.item()*100:.2f}%\n")
            f.write("-----------------------------------------\n")
            f.write("DETECÇÃO YOLO (Especialista):\n")
            f.write(f"  - Modelo Usado:  {yolo_filename}\n")
            f.write(f"  - Tempo Gasto:   {elapsed_time:.2f} segundos\n")
            
            # Conta quantidade de detecções se for imagem simples ou sumariza
            if is_image and len(results) > 0:
                boxes = results[0].boxes
                f.write(f"  - Detecções:     {len(boxes)} objeto(s) encontrado(s)\n")
                if len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        conf = box.conf.item()
                        cls = int(box.cls.item())
                        name = yolo_model.names.get(cls, str(cls))
                        f.write(f"    * Obj {i+1}: {name} (Conf: {conf*100:.2f}%)\n")
            f.write("=========================================\n")
            
        print(f"✓ Resumo da execução salvo em: {report_path}")
    except Exception as e:
        print(f"Aviso: Não foi possível salvar o arquivo de resumo: {e}")

    print(f"\nSucesso! Os resultados anotados estão salvos na pasta:\n  --> {save_dir}\n")

if __name__ == "__main__":
    main()
