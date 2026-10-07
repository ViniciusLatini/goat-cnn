#!/usr/bin/env python3
import os
import csv
from pathlib import Path

# Configuração de diretórios e nomes dos modelos especialistas
BASE_DIR = Path(__file__).resolve().parent
MODELS = {
    "yolo-8-front-dia": "YOLOv8 Frontal Dia",
    "yolo-8-front-noite": "YOLOv8 Frontal Noite",
    "yolo-8-sup-dia": "YOLOv8 Superior Dia",
    "yolo-8-sup-noite": "YOLOv8 Superior Noite"
}

def get_best_metrics_from_csv(csv_path):
    best_row = None
    best_map50 = -1.0
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Remove espaços extras dos nomes das colunas
            fieldnames_clean = [x.strip() for x in reader.fieldnames]
            
            # Identifica as colunas relevantes
            col_epoch = next((x for x in fieldnames_clean if 'epoch' in x), None)
            col_precision = next((x for x in fieldnames_clean if 'precision(B)' in x or 'precision' in x), None)
            col_recall = next((x for x in fieldnames_clean if 'recall(B)' in x or 'recall' in x), None)
            col_map50 = next((x for x in fieldnames_clean if 'mAP50(B)' in x or 'mAP50' in x), None)
            col_map50_95 = next((x for x in fieldnames_clean if 'mAP50-95(B)' in x or 'mAP50-95' in x), None)
            
            if not col_map50:
                print(f"  [!] Coluna mAP50 não encontrada em: {csv_path}")
                return None

            for row in reader:
                # Limpa chaves e valores da linha
                row_clean = {k.strip(): v.strip() for k, v in row.items()}
                try:
                    map50_val = float(row_clean[col_map50])
                    if map50_val > best_map50:
                        best_map50 = map50_val
                        best_row = {
                            "epoch": int(row_clean.get(col_epoch, 0)) if col_epoch else 0,
                            "precision": float(row_clean.get(col_precision, 0.0)) if col_precision else 0.0,
                            "recall": float(row_clean.get(col_recall, 0.0)) if col_recall else 0.0,
                            "map50": map50_val,
                            "map50_95": float(row_clean.get(col_map50_95, 0.0)) if col_map50_95 else 0.0
                        }
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        print(f"  [!] Erro ao processar {csv_path}: {e}")
        return None
        
    return best_row

def main():
    results = []
    
    print("Iniciando extração de métricas dos melhores folds...")
    
    for dir_name, display_name in MODELS.items():
        model_path = BASE_DIR / dir_name
        runs_dir = model_path / "runs" / "detect"
        if not runs_dir.exists() and (model_path / "runs" / "detect" / "runs" / "detect").exists():
            runs_dir = model_path / "runs" / "detect" / "runs" / "detect"
        
        if not runs_dir.exists():
            print(f"\n[!] Diretório de execuções não encontrado para {display_name}: {runs_dir}")
            continue
            
        print(f"\nAnalisando {display_name} em '{runs_dir}'...")
        folds_data = []
        
        # Encontra todas as subpastas fold_X
        for fold_dir in runs_dir.iterdir():
            if fold_dir.is_dir() and "fold" in fold_dir.name:
                csv_path = fold_dir / "results.csv"
                if csv_path.exists():
                    best_metrics = get_best_metrics_from_csv(csv_path)
                    if best_metrics:
                        best_metrics["fold"] = fold_dir.name
                        folds_data.append(best_metrics)
                        print(f"  -> {fold_dir.name}: mAP50 = {best_metrics['map50']:.4f} (Época {best_metrics['epoch']})")
        
        if not folds_data:
            print(f"  [!] Nenhum fold com dados válidos encontrado para {display_name}")
            continue
            
        # Seleciona o melhor fold baseado no maior mAP50
        best_fold = max(folds_data, key=lambda x: x['map50'])
        
        # Calcula F1-Score: 2 * (P * R) / (P + R)
        p = best_fold['precision']
        r = best_fold['recall']
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        best_fold['f1_score'] = f1
        best_fold['model_name'] = display_name
        
        results.append(best_fold)
        print(f"  => Vencedor: {best_fold['fold']} com mAP50 = {best_fold['map50']:.4f}")

    if not results:
        print("\n[!] Nenhuma métrica pôde ser extraída.")
        return

    # Gerar representações de tabelas
    # 1. Tabela Markdown
    md_lines = []
    md_lines.append("| Modelo | Melhor Fold | Época | Precisão (P) | Revocação (R) | F1-Score | mAP50 | mAP50-95 |")
    md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        md_lines.append(f"| {r['model_name']} | {r['fold']} | {r['epoch']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1_score']:.4f} | {r['map50']:.4f} | {r['map50_95']:.4f} |")
    
    markdown_table = "\n".join(md_lines)
    
    # 2. Tabela LaTeX
    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Métricas dos Melhores Folds dos Modelos YOLOv8 Especialistas}")
    latex_lines.append(r"\label{tab:yolo_especialistas}")
    latex_lines.append(r"\begin{tabular}{llcccccc}")
    latex_lines.append(r"\hline")
    latex_lines.append(r"Modelo & Fold & Época & Precisão (P) & Revocação (R) & F1-Score & mAP50 & mAP50-95 \\")
    latex_lines.append(r"\hline")
    for r in results:
        fold_escaped = r['fold'].replace('_', r'\_')
        latex_lines.append(f"{r['model_name']} & {fold_escaped} & {r['epoch']} & {r['precision']:.4f} & {r['recall']:.4f} & {r['f1_score']:.4f} & {r['map50']:.4f} & {r['map50_95']:.4f} \\\\")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    
    latex_table = "\n".join(latex_lines)

    # 3. CSV
    csv_lines = []
    csv_lines.append("Modelo,Melhor Fold,Epoca,Precisao,Revocacao,F1-Score,mAP50,mAP50-95")
    for r in results:
        csv_lines.append(f"{r['model_name']},{r['fold']},{r['epoch']},{r['precision']:.4f},{r['recall']:.4f},{r['f1_score']:.4f},{r['map50']:.4f},{r['map50_95']:.4f}")
    
    csv_table = "\n".join(csv_lines)

    # Exibe no terminal
    print("\n" + "="*30 + " TABELA MARKDOWN " + "="*30)
    print(markdown_table)
    
    print("\n" + "="*30 + " TABELA LATEX " + "="*30)
    print(latex_table)

    # Salva arquivos
    output_md_path = BASE_DIR / "tabela_resultados_yolo.md"
    output_tex_path = BASE_DIR / "tabela_resultados_yolo.tex"
    output_csv_path = BASE_DIR / "tabela_resultados_yolo.csv"
    
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_table)
    with open(output_tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    with open(output_csv_path, "w", encoding="utf-8") as f:
        f.write(csv_table)

    print(f"\n[+] Tabelas exportadas com sucesso:")
    print(f"  - Markdown: {output_md_path}")
    print(f"  - LaTeX: {output_tex_path}")
    print(f"  - CSV: {output_csv_path}")

if __name__ == "__main__":
    main()
