#!/usr/bin/env python3
"""
metric-k-fold.py

Analisa os resultados dos 5 folds de treinamento do YOLOv8 Overall
(runs/detect/runs/detect/fold_X/results.csv), extraindo, para cada fold, o
ponto de convergência de melhor mAP50 (Época, Precisão, Revocação, F1-Score
e mAP50-95), identifica o fold vencedor e copia automaticamente seus pesos
(weights/best.pt) para a raiz do projeto como best.pt.
"""

import csv
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR / "runs" / "detect" / "runs" / "detect"
OUTPUT_BEST_MODEL = BASE_DIR / "best.pt"


def get_best_metrics_from_csv(csv_path):
    """
    Lê o results.csv de um fold e retorna um dicionário com as métricas
    (época, precisão, revocação, mAP50, mAP50-95) do ponto de melhor mAP50
    observado ao longo do treinamento (early stopping / melhor checkpoint).
    """
    best_row = None
    best_map50 = -1.0

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames_clean = [x.strip() for x in reader.fieldnames]

            col_epoch = next((x for x in fieldnames_clean if 'epoch' in x), None)
            col_precision = next((x for x in fieldnames_clean if 'precision(B)' in x or x == 'precision'), None)
            col_recall = next((x for x in fieldnames_clean if 'recall(B)' in x or x == 'recall'), None)
            col_map50 = next((x for x in fieldnames_clean if 'mAP50(B)' in x or x == 'mAP50'), None)
            col_map50_95 = next((x for x in fieldnames_clean if 'mAP50-95(B)' in x or x == 'mAP50-95'), None)

            if not col_map50:
                print(f"  [!] Coluna mAP50 não encontrada em: {csv_path}")
                return None

            for row in reader:
                row_clean = {k.strip(): v.strip() for k, v in row.items()}
                try:
                    map50_val = float(row_clean[col_map50])
                except (ValueError, TypeError, KeyError):
                    continue

                if map50_val > best_map50:
                    best_map50 = map50_val
                    best_row = {
                        "epoch": int(float(row_clean.get(col_epoch, 0))) if col_epoch else 0,
                        "precision": float(row_clean.get(col_precision, 0.0)) if col_precision else 0.0,
                        "recall": float(row_clean.get(col_recall, 0.0)) if col_recall else 0.0,
                        "map50": map50_val,
                        "map50_95": float(row_clean.get(col_map50_95, 0.0)) if col_map50_95 else 0.0,
                    }
    except Exception as e:
        print(f"  [!] Erro ao processar {csv_path}: {e}")
        return None

    return best_row


def recover_and_save_best_model():
    print(f"\nAnalisando resultados dos folds em: '{PROJECT_DIR}'...")

    if not PROJECT_DIR.exists():
        print(f"Erro: A pasta '{PROJECT_DIR}' não existe. Rode o treinamento K-Fold primeiro.")
        return

    folds_data = []

    for fold_dir in sorted(PROJECT_DIR.iterdir()):
        if not (fold_dir.is_dir() and "fold" in fold_dir.name):
            continue

        csv_file = fold_dir / "results.csv"
        weights_file = fold_dir / "weights" / "best.pt"

        if not csv_file.exists():
            print(f"  [!] {fold_dir.name}: results.csv não encontrado, pulando.")
            continue
        if not weights_file.exists():
            print(f"  [!] {fold_dir.name}: weights/best.pt não encontrado, pulando.")
            continue

        metrics = get_best_metrics_from_csv(csv_file)
        if metrics is None:
            continue

        p = metrics["precision"]
        r = metrics["recall"]
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        metrics["f1_score"] = f1
        metrics["fold"] = fold_dir.name
        metrics["weights_path"] = weights_file

        folds_data.append(metrics)
        print(
            f"  -> {fold_dir.name}: Época={metrics['epoch']:>4} | "
            f"P={p:.4f} R={r:.4f} F1={f1:.4f} "
            f"mAP50={metrics['map50']:.4f} mAP50-95={metrics['map50_95']:.4f}"
        )

    if not folds_data:
        print("\n[!] Nenhum treinamento válido encontrado.")
        return

    best_fold = max(folds_data, key=lambda x: x["map50"])
    avg_map50 = sum(x["map50"] for x in folds_data) / len(folds_data)
    avg_map50_95 = sum(x["map50_95"] for x in folds_data) / len(folds_data)

    print("\n--- RESUMO DA VALIDAÇÃO CRUZADA (K-FOLD) ---")
    print(f"{'Fold':<10}{'Época':>8}{'Precisão':>12}{'Revocação':>12}{'F1-Score':>12}{'mAP50':>10}{'mAP50-95':>12}")
    for m in folds_data:
        print(
            f"{m['fold']:<10}{m['epoch']:>8}{m['precision']:>12.4f}{m['recall']:>12.4f}"
            f"{m['f1_score']:>12.4f}{m['map50']:>10.4f}{m['map50_95']:>12.4f}"
        )
    print(f"\nMédia mAP50    : {avg_map50:.4f}")
    print(f"Média mAP50-95 : {avg_map50_95:.4f}")

    print("\n--- FOLD VENCEDOR ---")
    print(f"Fold        : {best_fold['fold']}")
    print(f"Época       : {best_fold['epoch']}")
    print(f"Precisão    : {best_fold['precision']:.4f}")
    print(f"Revocação   : {best_fold['recall']:.4f}")
    print(f"F1-Score    : {best_fold['f1_score']:.4f}")
    print(f"mAP50       : {best_fold['map50']:.4f}")
    print(f"mAP50-95    : {best_fold['map50_95']:.4f}")
    print(f"Pesos       : {best_fold['weights_path']}")

    print(f"\nCopiando pesos do fold vencedor para '{OUTPUT_BEST_MODEL}'...")
    shutil.copy(best_fold["weights_path"], OUTPUT_BEST_MODEL)
    print("[OK] Cópia concluída com sucesso!")


if __name__ == "__main__":
    recover_and_save_best_model()
