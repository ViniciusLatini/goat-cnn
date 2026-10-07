#!/usr/bin/env python3
"""
run_parallel_kfold_trainings.py

Script para execução dos treinamentos dos especialistas YOLOv8.

IMPORTANTE: Com 1 GPU física (RTX 5070), rodar 2 processos em paralelo
causa CUDA Out of Memory na inicialização. Por isso, o padrão é --max-parallel 1
(sequencial), que maximiza o uso da VRAM por fold e é a configuração correta.

O --batch-size -1 (AutoBatch) é usado por padrão para que o YOLOv8 determine
automaticamente o maior batch que cabe na VRAM disponível (tipicamente 32-64
para o YOLOv8m na RTX 5070 com 12GB).

O dataset e o diretório raiz do projeto vivem em HDs mecânicos (não há SSD
disponível para o Linux nesta máquina), então o dataloading é o gargalo real,
não a GPU. Por isso o treino usa cache='ram' (cacheia as imagens decodificadas
em memória após a 1a época, eliminando I/O de disco no restante do treino) e
--workers mais alto que o default do Ultralytics, para aproveitar os 16 cores
da CPU e esconder a latência do disco.

Uso:
  python run_parallel_kfold_trainings.py
  python run_parallel_kfold_trainings.py --epochs 300
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SPECIALIST_DIRS = [
    'yolo-8-front-dia',
    'yolo-8-front-noite',
    'yolo-8-sup-dia',
    'yolo-8-sup-noite'
]

def clean_corrupted_caches():
    """
    Remove arquivos .cache antigos/corrompidos do dataset para evitar o erro _pickle.UnpicklingError.
    """
    ds_base = Path('/run/media/vini/061a1dd7-8e19-49bb-a45c-87c4603733ea/vini/ds')
    if ds_base.exists():
        for cache_file in ds_base.rglob('*.cache'):
            try:
                cache_file.unlink()
            except Exception:
                pass

def train_single_fold(specialist_dir, fold_idx, epochs, batch_size, imgsz, device, workers):
    """
    Executa o treinamento de um único fold em um processo isolado.
    """
    base_dir = Path(__file__).parent.absolute()
    target_folder = base_dir / specialist_dir
    yaml_file = target_folder / f"data_fold_{fold_idx}.yaml"
    pretrained_weights = base_dir / "weights" / "yolov8m.pt"

    if not yaml_file.exists():
        return (specialist_dir, fold_idx, False, f"Arquivo {yaml_file.name} não encontrado.")

    print(f"   [+] [START] {specialist_dir} | Fold {fold_idx + 1}/5 (Batch={batch_size}, Device={device}, Workers={workers})")

    project_dir = (target_folder / "runs" / "detect").absolute()

    # Comando inline do Python chamando a API do Ultralytics YOLO
    py_code = f"""
from ultralytics import YOLO

model = YOLO('{pretrained_weights}')
results = model.train(
    data='{yaml_file.absolute()}',
    epochs={epochs},
    imgsz={imgsz},
    batch={batch_size},
    patience=30,
    device='{device}',
    workers={workers},
    cache='ram',
    project='{project_dir}',
    name='fold_{fold_idx + 1}',
    exist_ok=True,
    verbose=False
)
"""
    try:
        res = subprocess.run(
            [sys.executable, "-c", py_code],
            cwd=target_folder,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
        print(f"   [✓] [FINISHED] {specialist_dir} | Fold {fold_idx + 1}/5 concluído com sucesso!")
        return (specialist_dir, fold_idx, True, "Sucesso")
    except subprocess.CalledProcessError as e:
        print(f"   [!] [ERROR] {specialist_dir} | Fold {fold_idx + 1}/5 falhou!")
        # Salvar log completo do erro na pasta do especialista
        error_log_path = target_folder / f"fold_{fold_idx + 1}_error.log"
        try:
            error_log_path.write_text(e.output, encoding="utf-8")
        except Exception:
            pass
        return (specialist_dir, fold_idx, False, e.output)

def main():
    parser = argparse.ArgumentParser(description="Treinamento Paralelo de Folds YOLOv8 na GPU")
    parser.add_argument("--max-parallel", type=int, default=1,
                        help="Treinos em paralelo. DEVE ser 1 para 1 GPU física. Valor > 1 causa CUDA OOM.")
    parser.add_argument("--batch-size", type=int, default=-1,
                        help="Batch size. -1 = AutoBatch (YOLOv8 ajusta automaticamente ao max. da VRAM).")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Número de épocas por fold (Default: 300)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Tamanho das imagens (Default: 640)")
    parser.add_argument("--device", type=str, default="0",
                        help="Dispositivo GPU CUDA (Default: 0)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Workers do dataloader (Default: 8). Maquina tem 16 cores; "
                             "o dataset/HD original é mecânico, entao mais workers ajuda "
                             "a esconder a latência de I/O.")
    parser.add_argument("--specialists", nargs="+", default=SPECIALIST_DIRS,
                        help="Especialistas para treinar")
    args = parser.parse_args()

    clean_corrupted_caches()

    if args.max_parallel > 1:
        print("[AVISO] --max-parallel > 1 com 1 GPU física causará CUDA Out of Memory.")
        print("        Usando --max-parallel 1 (sequencial) automaticamente.")
        args.max_parallel = 1

    batch_display = "AutoBatch" if args.batch_size == -1 else str(args.batch_size)
    print("==================================================================")
    print("    TREINAMENTO SEQUENCIAL DOS ESPECIALISTAS YOLOV8 (1 GPU)       ")
    print("==================================================================")
    print(f" GPU Alvo          : CUDA {args.device}")
    print(f" Modo              : Sequencial (1 fold por vez, maximiza VRAM)")
    print(f" Batch Size / Fold : {batch_display}")
    print(f" Épocas / Fold     : {args.epochs}")
    print(f" Workers / Fold    : {args.workers}")
    print(f" Cache             : RAM (dataset cacheado em memória após a 1a época)")
    print(f" Especialistas     : {args.specialists}")
    print("==================================================================")

    # Montar fila de tarefas (20 folds no total para 4 especialistas)
    tasks = []
    for spec_dir in args.specialists:
        for fold in range(5):
            tasks.append((spec_dir, fold))

    print(f"Total de jobs na fila: {len(tasks)} folds.")
    print(f"Iniciando pool com {args.max_parallel} workers em paralelo...\n")

    completed_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(
                train_single_fold,
                spec_dir,
                fold_idx,
                args.epochs,
                args.batch_size,
                args.imgsz,
                args.device,
                args.workers
            ): (spec_dir, fold_idx)
            for spec_dir, fold_idx in tasks
        }

        for future in as_completed(futures):
            spec_dir, fold_idx, success, msg = future.result()
            if success:
                completed_count += 1
            else:
                failed_count += 1
                msg_lines = msg.strip().splitlines()
                last_lines = "\n".join(msg_lines[-20:]) if len(msg_lines) > 20 else msg
                print(f"Detalhamento do Erro ({spec_dir} fold {fold_idx + 1}):\n{last_lines}")

    print("\n=================================================================")
    print(f" RESUMO DA EXECUÇÃO PARALELA")
    print(f" Treinos concluídos com SUCESSO: {completed_count}/{len(tasks)}")
    print(f" Treinos com FALHA             : {failed_count}/{len(tasks)}")
    print("=================================================================")

if __name__ == '__main__':
    main()
