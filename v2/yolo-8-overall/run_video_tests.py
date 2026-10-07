#!/usr/bin/env python3
"""
run_video_tests.py

Script único para validar o modelo recém-treinado (best.pt) nos vídeos de
./testes:

  1. Verifica se algum vídeo de teste é um vazamento de sessão de treino
     (check_session_leakage.py: pré-filtro por aHash + confirmação por
     MSE pixel-a-pixel contra as imagens usadas nos folds K-Fold).
  2. Roda a inferência YOLO em cada vídeo (vídeo anotado + labels .txt).
  3. Salva tudo (vídeos anotados, labels, relatório de vazamento e
     relatório de inferência) em uma pasta de saída única e com timestamp,
     dentro de ./output.

Uso:
    python run_video_tests.py
    python run_video_tests.py --model best.pt --source ./testes --conf 0.7 --iou 0.7
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

import check_session_leakage as leakage

BASE_DIR = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Roda a checagem de vazamento de sessão + inferência YOLO nos vídeos de teste, "
                    "salvando tudo em uma pasta de saída com timestamp."
    )
    parser.add_argument("--model", type=str, default="best.pt", help="Pesos do modelo treinado (padrão: best.pt).")
    parser.add_argument("--source", type=str, default="./testes", help="Pasta com os vídeos de teste (padrão: ./testes).")
    parser.add_argument("--output-root", type=str, default="./output", help="Pasta raiz onde a pasta da execução será criada (padrão: ./output).")
    parser.add_argument("--conf", type=float, default=0.5, help="Limiar de confiança (padrão: 0.5).")
    parser.add_argument("--iou", type=float, default=0.7, help="Limiar de NMS IoU (padrão: 0.7).")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho de entrada (padrão: 640).")
    parser.add_argument("--device", type=str, default="", help="'cuda', '0' ou 'cpu'. Vazio = detecção automática.")
    parser.add_argument("--skip-leakage-check", action="store_true", help="Pula a checagem de vazamento de sessão.")
    return parser.parse_args()


def get_video_files(source_path: Path):
    if source_path.is_file():
        return [source_path] if source_path.suffix.lower() in VIDEO_EXTENSIONS else []
    if not source_path.exists() or not source_path.is_dir():
        return []
    return [f for f in sorted(source_path.iterdir()) if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]


def run_leakage_check(run_dir: Path):
    print("\n" + "=" * 70)
    print(" ETAPA 1/2: CHECAGEM DE VAZAMENTO DE SESSÃO (TESTE x TREINO)")
    print("=" * 70)
    report_path = run_dir / "leakage_report.txt"
    sessions = leakage.load_training_sessions()
    leakage.index_session_images(sessions)
    videos = sorted(leakage.TESTES_DIR.glob("*.avi")) + sorted(leakage.TESTES_DIR.glob("*.mp4"))

    results = []
    for v in videos:
        print(f"\n>>> Analisando: {v.name}")
        res = leakage.check_video(v, sessions)
        results.append(res)
        status = "VAZAMENTO CONFIRMADO" if res["is_leak"] else "SEM EVIDÊNCIA DE VAZAMENTO"
        print(f"  Data coincide com treino: {'sim' if res['date_matches_training'] else 'não'} | "
              f"MSE mínimo: {res['best_mse']} | Status: {status}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(" VERIFICAÇÃO DE VAZAMENTO DE SESSÃO (VÍDEOS DE TESTE x TREINO)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Sessões de treino: {len(sessions)}\n")
        f.write(f"Pré-filtro aHash (Hamming <= {leakage.HAMMING_PREFILTER_THRESHOLD}), "
                f"confirmação por MSE pixel-a-pixel (<= {leakage.MSE_DUPLICATE_THRESHOLD} = duplicata confirmada)\n")
        f.write("Referência: MSE self-vs-self = 0; MSE entre frames DIFERENTES da MESMA sessão de treino ~750-1200.\n")
        f.write("-" * 70 + "\n")
        any_leak = False
        for r in results:
            status = "VAZAMENTO CONFIRMADO" if r["is_leak"] else "SEM EVIDÊNCIA DE VAZAMENTO"
            any_leak = any_leak or r["is_leak"]
            f.write(f"\n* Vídeo: {r['video']}\n")
            f.write(f"  - Data extraída do nome: {r['parsed_date']}\n")
            f.write(f"  - Data coincide com sessão de treino: {'sim' if r['date_matches_training'] else 'não'}\n")
            f.write(f"  - Sessões comparadas: {r['n_candidate_sessions']}\n")
            f.write(f"  - Menor MSE (pixel) encontrado: {r['best_mse']}\n")
            f.write(f"  - Imagem de treino mais próxima: {r['best_match_session']} / {r['best_match_image']}\n")
            f.write(f"  - Status: {status}\n")
        f.write("\n" + "-" * 70 + "\n")
        f.write(f"CONCLUSÃO GERAL: {'HÁ VAZAMENTO CONFIRMADO EM AO MENOS 1 VÍDEO' if any_leak else 'Nenhum vídeo de teste é uma duplicata confirmada de frames de treino.'}\n")
        f.write("=" * 70 + "\n")

    print(f"\nRelatório de vazamento salvo em: {report_path.resolve()}")
    return results, report_path


def run_inference(model_path: Path, video_files, output_dir: Path, conf, iou, imgsz, device_arg):
    print("\n" + "=" * 70)
    print(" ETAPA 2/2: INFERÊNCIA YOLO NOS VÍDEOS DE TESTE")
    print("=" * 70)

    if not model_path.exists():
        print(f"\n[ERRO] Modelo não encontrado em: '{model_path.resolve()}'")
        sys.exit(1)

    device = device_arg or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Modelo: {model_path} | Dispositivo: {device.upper()}")
    if "cuda" in device and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(str(model_path))

    total_start = time.time()
    reports = []
    for i, video in enumerate(video_files, 1):
        print(f"\n>>> Processando [{i}/{len(video_files)}]: {video.name}")
        start = time.time()
        try:
            results = model.predict(
                source=str(video),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                save=True,
                save_txt=True,
                save_conf=True,
                # Caminhos absolutos: com caminho relativo o Ultralytics redireciona
                # `project` para o runs_dir global configurado (~/.config/Ultralytics/settings.json).
                project=str(output_dir.resolve().parent),
                name=output_dir.resolve().name,
                exist_ok=True,
                verbose=False,
            )
            elapsed = time.time() - start
            num_frames = len(results) if isinstance(results, list) else 0
            fps = (num_frames / elapsed) if elapsed > 0 and num_frames > 0 else 0
            total_detections = sum(len(r.boxes) for r in results) if results else 0

            print(f"  -> OK | Frames: {num_frames} | {elapsed:.2f}s (~{fps:.1f} FPS) | Detecções: {total_detections}")
            reports.append({
                "video": video.name, "frames": num_frames, "tempo_seg": elapsed,
                "fps": fps, "deteccoes": total_detections, "status": "OK",
            })
        except Exception as e:
            print(f"  -> [ERRO ao processar {video.name}]: {e}")
            reports.append({"video": video.name, "status": f"ERRO: {e}"})

    total_elapsed = time.time() - total_start
    return reports, total_elapsed, device, model_path


def write_inference_report(report_path, reports, total_elapsed, device, model_path, source_path, output_dir, conf, iou):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("        RELATÓRIO DE TESTE DE VÍDEOS (YOLO)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modelo Utilizado: {model_path.resolve()}\n")
        f.write(f"Dispositivo: {device}\n")
        f.write(f"Pasta de Origem: {source_path.resolve()}\n")
        f.write(f"Pasta de Destino: {output_dir.resolve()}\n")
        f.write(f"Limiar de Confiança: {conf} | IoU: {iou}\n")
        f.write(f"Tempo Total de Execução: {total_elapsed:.2f}s\n")
        f.write("-" * 60 + "\n")
        f.write("RESUMO POR VÍDEO:\n")
        for rep in reports:
            if rep["status"] == "OK":
                f.write(f"\n* Vídeo: {rep['video']}\n")
                f.write(f"  - Status: Concluído\n")
                f.write(f"  - Frames: {rep['frames']}\n")
                f.write(f"  - Duração: {rep['tempo_seg']:.2f}s ({rep['fps']:.1f} FPS)\n")
                f.write(f"  - Total de Caixas Detectadas: {rep['deteccoes']}\n")
            else:
                f.write(f"\n* Vídeo: {rep['video']} -> {rep['status']}\n")
        f.write("=" * 60 + "\n")


def main():
    args = parse_args()
    source_path = Path(args.source)
    model_path = Path(args.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"test_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("   SUÍTE DE TESTE: VAZAMENTO DE SESSÃO + INFERÊNCIA YOLO")
    print("=" * 70)
    print(f"Pasta de saída desta execução: {run_dir.resolve()}")

    if not args.skip_leakage_check:
        run_leakage_check(run_dir)
    else:
        print("\n[INFO] Checagem de vazamento de sessão pulada (--skip-leakage-check).")

    video_files = get_video_files(source_path)
    if not video_files:
        print(f"\n[AVISO] Nenhum vídeo encontrado em: '{source_path}'")
        sys.exit(1)

    reports, total_elapsed, device, model_path = run_inference(
        model_path, video_files, run_dir, args.conf, args.iou, args.imgsz, args.device
    )

    report_path = run_dir / "relatorio_teste.txt"
    write_inference_report(report_path, reports, total_elapsed, device, model_path, source_path, run_dir, args.conf, args.iou)

    print("\n" + "=" * 70)
    print("                 PROCESSAMENTO FINALIZADO!")
    print("=" * 70)
    print(f"Tempo total: {total_elapsed:.2f}s")
    print(f"Resultados salvos em: {run_dir.resolve()}")
    print(f"  - Relatório de inferência: {report_path.name}")
    if not args.skip_leakage_check:
        print(f"  - Relatório de vazamento: leakage_report.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
