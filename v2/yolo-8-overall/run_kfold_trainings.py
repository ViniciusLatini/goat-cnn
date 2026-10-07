#!/usr/bin/env python3
"""
run_kfold_trainings.py

Script de execução do treinamento K-Fold (5 folds) do modelo YOLOv8 Overall
(baseline monolítico generalista), usando os splits temporais gerados por
generate_temporal_kfold.py (data_fold_0.yaml ... data_fold_4.yaml).

IMPORTANTE: Com 1 GPU física (RTX 5070, 12GB VRAM), os 5 folds são treinados
SEQUENCIALMENTE, um de cada vez, para maximizar a VRAM disponível por fold e
evitar CUDA Out of Memory. Não rode folds em paralelo nesta máquina.

O --batch -1 (AutoBatch) é usado por padrão para que o YOLOv8 determine
automaticamente o maior batch que cabe na VRAM disponível.

O dataset vive em um HD mecânico (não há SSD disponível para o Linux nesta
máquina), então o dataloading é o gargalo real, não a GPU. Por isso o treino
usa cache='ram' (cacheia as imagens decodificadas em memória após a 1a época,
eliminando I/O de disco no restante do treino) e workers=8 para aproveitar
os núcleos de CPU e esconder a latência do disco.

Uso:
  python run_kfold_trainings.py
  python run_kfold_trainings.py --epochs 300 --folds 0 1 2 3 4
"""

import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = Path(
    '/run/media/vini/061a1dd7-8e19-49bb-a45c-87c4603733ea/vini/ds-overall-raw'
)
DEFAULT_PRETRAINED_WEIGHTS = BASE_DIR / "yolov8m.pt"
DEFAULT_PROJECT_DIR = BASE_DIR / "runs" / "detect" / "runs" / "detect"

NUM_FOLDS = 5


def clean_corrupted_caches(dataset_path):
    """
    Remove arquivos .cache antigos/corrompidos do dataset e do próprio
    diretório do projeto (o Ultralytics cria um .cache ao lado das listas
    train_fold_X.txt/val_fold_X.txt) para evitar o erro
    _pickle.UnpicklingError ao reaproveitar caches de um split anterior
    (ex: do K-Fold aleatório antigo).
    """
    print("Limpando caches antigos (.cache) antes de iniciar os treinos...")
    removed = 0
    search_dirs = [Path(dataset_path), BASE_DIR]
    for base in search_dirs:
        if not base.exists():
            continue
        for cache_file in base.rglob('*.cache'):
            try:
                cache_file.unlink()
                removed += 1
            except Exception as e:
                print(f"  [!] Não foi possível remover {cache_file}: {e}")
    print(f"  [OK] {removed} arquivo(s) .cache removido(s).")


def train_single_fold(fold_idx, yaml_file, pretrained_weights, project_dir,
                       epochs, batch_size, imgsz, device, workers):
    """
    Executa o treinamento de um único fold via API do Ultralytics.
    """
    from ultralytics import YOLO

    run_name = f"fold_{fold_idx + 1}"
    print("\n=================================================================")
    print(f" Iniciando Fold {fold_idx + 1}/{NUM_FOLDS}")
    print(f" YAML       : {yaml_file}")
    print(f" Pesos base : {pretrained_weights}")
    print(f" Projeto    : {project_dir}")
    print(f" Nome       : {run_name}")
    print(f" Batch      : {'AutoBatch' if batch_size == -1 else batch_size}")
    print(f" Device     : {device}")
    print("=================================================================")

    model = YOLO(str(pretrained_weights))
    results = model.train(
        data=str(yaml_file),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=30,
        device=device,
        workers=workers,
        cache='ram',
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )
    print(f" [OK] Fold {fold_idx + 1}/{NUM_FOLDS} concluído. Resultados em: {results.save_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Treinamento Sequencial dos 5 Folds Temporais do YOLOv8 Overall (1 GPU)"
    )
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH),
                         help="Caminho do dataset (usado apenas para limpeza de caches)")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_PRETRAINED_WEIGHTS),
                         help="Pesos pré-treinados usados como ponto de partida em cada fold")
    parser.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR),
                         help="Diretório raiz onde os resultados de cada fold serão salvos")
    parser.add_argument("--batch", type=int, default=-1,
                         help="Batch size. -1 = AutoBatch (ajusta automaticamente ao max. da VRAM).")
    parser.add_argument("--epochs", type=int, default=300,
                         help="Número máximo de épocas por fold (Default: 300)")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Tamanho das imagens (Default: 640)")
    parser.add_argument("--device", type=str, default="cuda:0",
                         help="Dispositivo CUDA (Default: cuda:0)")
    parser.add_argument("--workers", type=int, default=8,
                         help="Workers do dataloader (Default: 8)")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(NUM_FOLDS)),
                         help="Índices dos folds (0 a 4) a treinar. Default: todos.")
    args = parser.parse_args()

    clean_corrupted_caches(args.dataset_path)

    project_dir = Path(args.project_dir)
    pretrained_weights = Path(args.weights)

    print("==================================================================")
    print("    TREINAMENTO SEQUENCIAL K-FOLD - YOLOv8 OVERALL (1 GPU)        ")
    print("==================================================================")
    print(f" GPU Alvo          : {args.device}")
    print(f" Modo              : Sequencial (1 fold por vez, maximiza VRAM)")
    print(f" Batch Size / Fold : {'AutoBatch' if args.batch == -1 else args.batch}")
    print(f" Épocas / Fold     : {args.epochs}")
    print(f" Workers / Fold    : {args.workers}")
    print(f" Cache             : RAM (dataset cacheado em memória após a 1a época)")
    print(f" Folds a treinar   : {[f + 1 for f in args.folds]}")
    print("==================================================================")

    completed, failed = [], []
    for fold_idx in args.folds:
        yaml_file = BASE_DIR / f"data_fold_{fold_idx}.yaml"
        if not yaml_file.exists():
            print(f"[!] Arquivo {yaml_file.name} não encontrado. Rode generate_temporal_kfold.py primeiro. Pulando fold {fold_idx}.")
            failed.append(fold_idx)
            continue
        try:
            train_single_fold(
                fold_idx, yaml_file, pretrained_weights, project_dir,
                args.epochs, args.batch, args.imgsz, args.device, args.workers,
            )
            completed.append(fold_idx)
        except Exception as e:
            print(f"[!] ERRO no fold {fold_idx}: {e}")
            failed.append(fold_idx)

    print("\n=================================================================")
    print(" RESUMO DA EXECUÇÃO")
    print(f" Folds concluídos com SUCESSO: {[f + 1 for f in completed]}")
    print(f" Folds com FALHA             : {[f + 1 for f in failed]}")
    print("=================================================================")

    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
