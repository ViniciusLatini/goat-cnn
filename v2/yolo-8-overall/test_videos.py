import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import torch
from ultralytics import YOLO

# Extensões de vídeo suportadas
VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}

def parse_args():
    parser = argparse.ArgumentParser(description="Script para testar vídeos usando modelo YOLO e salvar em pasta separada de output.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="best.pt", 
        help="Caminho para os pesos do modelo treinado (padrão: best.pt)"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default="./testes", 
        help="Pasta contendo os vídeos de teste ou caminho para um vídeo específico (padrão: ./testes)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./output", 
        help="Pasta de destino onde os vídeos processados e arquivos serão salvos (padrão: ./output)"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.5, 
        help="Limiar de confiança para detecção (padrão: 0.5)"
    )
    parser.add_argument(
        "--iou", 
        type=float, 
        default=0.7, 
        help="Limiar de NMS IoU (padrão: 0.7)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640, 
        help="Tamanho da imagem de entrada no modelo (padrão: 640)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="", 
        help="Dispositivo de execução ('cuda', '0', 'cpu'). Se omitido, detecta automaticamente."
    )
    parser.add_argument(
        "--no-save-txt", 
        action="store_true", 
        help="Não salvar arquivos .txt com coordenadas das predições (por padrão, o .txt é salvo)"
    )
    parser.add_argument(
        "--show", 
        action="store_true", 
        help="Exibir janela com o vídeo e predições em tempo real"
    )
    parser.add_argument(
        "--hide-labels", 
        action="store_true", 
        help="Ocultar os nomes das classes nas caixas delimitadoras"
    )
    parser.add_argument(
        "--hide-conf", 
        action="store_true", 
        help="Ocultar os valores de confiança nas caixas delimitadoras"
    )
    return parser.parse_args()


def get_video_files(source_path: Path):
    if source_path.is_file():
        if source_path.suffix.lower() in VIDEO_EXTENSIONS:
            return [source_path]
        else:
            return []
    
    if not source_path.exists() or not source_path.is_dir():
        return []

    video_files = [
        f for f in sorted(source_path.iterdir())
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return video_files


def main():
    args = parse_args()

    model_path = Path(args.model)
    source_path = Path(args.source)
    output_dir = Path(args.output)
    save_txt = not args.no_save_txt

    # Garante que a pasta de output exista
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("        INICIANDO TESTES DE VÍDEOS COM YOLO")
    print("=" * 65)

    # 1. Validação do Modelo
    if not model_path.exists():
        print(f"\n[ERRO] Modelo não encontrado em: '{model_path.resolve()}'")
        print("Certifique-se de que o arquivo 'best.pt' ou o caminho especificado existe.")
        sys.exit(1)

    print(f"Modelo carregado: {model_path}")

    # 2. Definição do Dispositivo (CUDA / CPU)
    if args.device:
        device = args.device
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"Dispositivo em uso: {device.upper()}")
    if 'cuda' in device and torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    # 3. Busca de Vídeos
    video_files = get_video_files(source_path)
    if not video_files:
        print(f"\n[AVISO] Nenhum vídeo ({', '.join(VIDEO_EXTENSIONS)}) encontrado em: '{source_path}'")
        sys.exit(1)

    print(f"\nVídeos encontrados ({len(video_files)}):")
    for idx, vf in enumerate(video_files, 1):
        file_size_mb = vf.stat().st_size / (1024 * 1024)
        print(f"  [{idx}] {vf.name} ({file_size_mb:.2f} MB)")

    print("-" * 65)
    print("Configurações:")
    print(f"  - Pasta de Saída (Output): {output_dir.resolve()}")
    print(f"  - Confiança mínima (conf): {args.conf}")
    print(f"  - IoU (NMS): {args.iou}")
    print(f"  - Tamanho da imagem (imgsz): {args.imgsz}")
    print(f"  - Salvar vídeos anotados: Sim")
    print(f"  - Salvar coordenadas TXT: {'Sim' if save_txt else 'Não'}")
    print("-" * 65)

    # 4. Inicializa o Modelo YOLO
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o modelo YOLO: {e}")
        sys.exit(1)

    # 5. Execução da Inferência Vídeo por Vídeo
    total_start_time = time.time()
    execution_reports = []
    
    for i, video in enumerate(video_files, 1):
        print(f"\n>>> Processando [{i}/{len(video_files)}]: {video.name}")
        start_time = time.time()

        try:
            results = model.predict(
                source=str(video),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                save=True,
                save_txt=save_txt,
                save_conf=save_txt,
                show=args.show,
                hide_labels=args.hide_labels,
                hide_conf=args.hide_conf,
                project=str(output_dir.parent),
                name=output_dir.name,
                exist_ok=True,  # Salva direto dentro da pasta output informada
                verbose=False
            )

            elapsed = time.time() - start_time
            num_frames = len(results) if isinstance(results, list) else 0
            fps = (num_frames / elapsed) if elapsed > 0 and num_frames > 0 else 0
            total_detections = sum(len(r.boxes) for r in results) if results else 0

            print(f"  -> Concluído com sucesso!")
            print(f"  -> Frames processados: {num_frames} | Tempo: {elapsed:.2f}s (~{fps:.1f} FPS)")
            print(f"  -> Total de detecções encontradas: {total_detections}")

            execution_reports.append({
                "video": video.name,
                "frames": num_frames,
                "tempo_seg": elapsed,
                "fps": fps,
                "deteccoes": total_detections,
                "status": "OK"
            })

        except Exception as e:
            print(f"  -> [ERRO ao processar {video.name}]: {e}")
            execution_reports.append({
                "video": video.name,
                "status": f"ERRO: {e}"
            })

    total_elapsed = time.time() - total_start_time

    # 6. Criação do Relatório de Execução em TXT na pasta de output
    report_file = output_dir / "relatorio_teste.txt"
    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("        RELATÓRIO DE TESTE DE VÍDEOS (YOLO)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Modelo Utilizado: {model_path.resolve()}\n")
            f.write(f"Pasta de Origem: {source_path.resolve()}\n")
            f.write(f"Pasta de Destino: {output_dir.resolve()}\n")
            f.write(f"Limiar de Confiança: {args.conf} | IoU: {args.iou}\n")
            f.write(f"Tempo Total de Execução: {total_elapsed:.2f}s\n")
            f.write("-" * 60 + "\n")
            f.write("RESUMO POR VÍDEO:\n")
            for rep in execution_reports:
                if rep["status"] == "OK":
                    f.write(f"\n* Vídeo: {rep['video']}\n")
                    f.write(f"  - Status: Concluído\n")
                    f.write(f"  - Frames: {rep['frames']}\n")
                    f.write(f"  - Duração: {rep['tempo_seg']:.2f}s ({rep['fps']:.1f} FPS)\n")
                    f.write(f"  - Total de Caixas Detectadas: {rep['deteccoes']}\n")
                else:
                    f.write(f"\n* Vídeo: {rep['video']} -> {rep['status']}\n")
            f.write("=" * 60 + "\n")
    except Exception as e:
        print(f"[AVISO] Não foi possível salvar o relatório: {e}")

    print("\n" + "=" * 65)
    print("                 PROCESSAMENTO FINALIZADO!")
    print("=" * 65)
    print(f"Tempo total gasto: {total_elapsed:.2f}s")
    print(f"Vídeos anotados, labels e relatório salvos em:")
    print(f"  📁 {output_dir.resolve()}")
    if report_file.exists():
        print(f"  📄 Relatório gerado: {report_file.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
