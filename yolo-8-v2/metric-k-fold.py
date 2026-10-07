import csv
import shutil
from pathlib import Path

PROJECT_DIR = Path("./runs/detect/kfold") 
OUTPUT_BEST_MODEL = "best.pt"

def get_best_metric_from_csv(csv_path):
    best_map = -1.0
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = [x.strip() for x in reader.fieldnames]
            
            map_col = next((x for x in fieldnames if 'mAP50(B)' in x or 'mAP50' in x), None)
            
            if not map_col:
                return 0.0

            for row in reader:
                row_clean = {k.strip(): v.strip() for k, v in row.items()}
                try:
                    val = float(row_clean[map_col])
                    if val > best_map:
                        best_map = val
                except ValueError:
                    continue
                    
    except Exception as e:
        print(f"Erro ao ler {csv_path}: {e}")
        return 0.0
        
    return best_map

def recover_and_save_best_model():
    print(f"\n Analisando resultados na pasta: '{PROJECT_DIR}'...")
    
    if not PROJECT_DIR.exists():
        print(f"Erro: A pasta '{PROJECT_DIR}' não existe. Verifique o caminho.")
        return

    folds_found = []
    
    # 1. Varre subpastas (fold_1, fold_2, etc.)
    for fold_dir in PROJECT_DIR.iterdir():
        if fold_dir.is_dir() and "fold" in fold_dir.name:
            csv_file = fold_dir / "results.csv"
            weights_file = fold_dir / "weights" / "best.pt"
            
            if csv_file.exists() and weights_file.exists():
                score = get_best_metric_from_csv(csv_file)
                
                folds_found.append({
                    "fold": fold_dir.name,
                    "score": score,
                    "path": weights_file
                })
                print(f"-> {fold_dir.name}: mAP50 encontrado = {score:.4f}")

    if not folds_found:
        print("Nenhum treinamento válido encontrado.")
        return


    best_run = max(folds_found, key=lambda x: x['score'])
    
    print("\n--- RESULTADO RECUPERADO ---")
    print(f"Vencedor: {best_run['fold']}")
    print(f"Melhor mAP50: {best_run['score']:.4f}")
    print(f"Caminho original: {best_run['path']}")

    # 3. Salva o arquivo
    print(f"\n Copiando para '{OUTPUT_BEST_MODEL}'...")
    shutil.copy(best_run['path'], OUTPUT_BEST_MODEL)
    print("Cópia concluída com sucesso!")

if __name__ == "__main__":
    recover_and_save_best_model()