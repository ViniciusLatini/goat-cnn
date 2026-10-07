#!/usr/bin/env python3
"""
generate_temporal_kfold.py

Script para gerar divisões de Validação Cruzada K-Fold (5 folds) para o modelo
YOLOv8 Overall (baseline monolítico generalista), lendo diretamente do
diretório montado do dataset consolidado bruto:
/run/media/vini/061a1dd7-8e19-49bb-a45c-87c4603733ea/vini/ds-overall-raw
(layout plano: images/ + labels/, sem split train/valid prévio)

Também suporta, via --dataset-path, o layout legado com subpastas
train/images + valid/images (ex: dataset-overall), usado nas rodadas
anteriores do TCC.

Estratégia (idêntica à usada nos especialistas do MoE, ver
/home/vini/tcc/moe/generate_temporal_kfold.py):

1. Consolida todas as imagens do dataset de origem. Se o dataset tiver o
   layout legado train/images + valid/images (o split train/valid original
   do Roboflow intercala frames do mesmo vídeo entre treino e validação, o
   que por si só já é uma fonte de vazamento temporal), os dois diretórios
   são unificados aqui e redivididos. Se o dataset tiver o layout plano
   images/ + labels/ (dataset bruto consolidado), todas as imagens de
   images/ são usadas diretamente.
2. Agrupa imagens por vídeo/sessão de gravação (via parsing do nome do
   arquivo).
3. Descarta imagens cujo nome não bate com nenhum padrão de vídeo/frame
   conhecido (não é possível garantir sua posição cronológica real, então não
   podem ser posicionadas com segurança em um bloco temporal contíguo).
4. Ordena os frames cronologicamente por índice de frame dentro de cada
   vídeo.
5. Aplica Bloqueio Temporal Contínuo (Contiguous Time-Block Split / Temporal
   K-Fold):
   - Cada fold recebe uma janela temporal contínua de 20% do vídeo para
     validação.
   - Os outros 80% (4 blocos restantes) são usados no treino.
6. Aplica uma zona de purga (purge/embargo buffer) de N frames em ambos os
   lados de cada bloco de validação: frames dentro dessa janela são
   descartados do fold (não entram nem em treino nem em validação), pois são
   temporalmente próximos o suficiente do bloco de validação para serem
   quase-duplicatas visuais dele. Isso mitiga (mas não elimina) o vazamento
   por adjacência temporal entre treino e validação.
7. Garante 0% de sobreposição *exata* de arquivos entre treino e validação.

Observação importante: o purge buffer reduz o vazamento por adjacência
temporal (frames quase-idênticos na fronteira do bloco), mas NÃO elimina o
vazamento de cena/sessão (mesmo fundo, mesmos indivíduos, mesma câmera)
quando poucas sessões de vídeo dominam o dataset. Ver o relatório de
auditoria gerado (kfold_audit_report.txt) para a contagem de frames por
vídeo/sessão.
"""

import re
import argparse
from pathlib import Path
from collections import defaultdict

NUM_FOLDS = 5
DEFAULT_PURGE_FRAMES = 15
DEFAULT_DATASET_PATH = Path(
    '/run/media/vini/061a1dd7-8e19-49bb-a45c-87c4603733ea/vini/ds-overall-raw'
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

NUM_CLASSES = 1
CLASS_NAMES = ['goat']


def parse_filename_info(filepath):
    """
    Extrai o ID do vídeo/sessão e o índice do frame a partir do nome da
    imagem.

    Retorna uma tupla (video_id: str, frame_idx: int), ou None quando o nome
    não bate com nenhum padrão conhecido — nesse caso não há garantia de que
    a imagem possa ser posicionada corretamente na sequência cronológica do
    vídeo, e ela é excluída do split (ver auditoria).

    Dois estilos de nome são suportados:
      a) Exportado pelo Roboflow, com hash ".rf.<hash>" antes da extensão
         real (a extensão original vira um sufixo literal "_jpg" embutido
         no nome, ex: "video-0000_jpg.rf.<hash>.jpg"). O hash é removido
         antes do parsing.
      b) Frame bruto extraído diretamente do vídeo, sem hash do Roboflow
         (ex: "video-frame_0000.jpg"). A extensão real é removida via
         Path.stem antes do parsing.

    Padrões reconhecidos (após normalização acima; o sufixo "_jpg" herdado
    do estilo Roboflow é opcional em todos eles):
      1. <video_prefix>_frame_<frame_num>[_jpg]
      2. <video_prefix>-frame_<frame_num>[_jpg]
      3. <video_prefix>-<frame_num>[_jpg]
      4. <video_prefix>_random_<frame_num>[_jpg]
      5. frame_<frame_num>[_jpg]
    """
    name = Path(filepath).name
    if '.rf.' in name:
        # Remove hash do roboflow (.rf.xxxx.jpg); o que sobra já termina
        # com o sufixo "_jpg" da extensão original codificada pelo Roboflow.
        base = name.split('.rf.')[0]
    else:
        # Sem hash do Roboflow: remove a extensão real do arquivo.
        base = Path(name).stem

    # 1. Padrão: <video_prefix>_frame_<frame_num>[_jpg]
    m = re.match(r'^(.*?)_frame_(\d+)(?:_jpg)?$', base)
    if m:
        return m.group(1), int(m.group(2))

    # 2. Padrão: <video_prefix>-frame_<frame_num>[_jpg] (frame bruto, sem hash)
    m = re.match(r'^(.*?)-frame_(\d+)(?:_jpg)?$', base)
    if m:
        return m.group(1), int(m.group(2))

    # 3. Padrão: <video_prefix>-<frame_num>[_jpg]
    m = re.match(r'^(.*?)-(\d+)(?:_jpg)?$', base)
    if m:
        return m.group(1), int(m.group(2))

    # 4. Padrão: <video_prefix>_random_<frame_num>[_jpg]
    m = re.match(r'^(.*?)_random_(\d+)(?:_jpg)?$', base)
    if m:
        return m.group(1), int(m.group(2))

    # 5. Padrão: frame_(\d+)[_jpg] (sequência sem prefixo de câmera/sessão)
    m = re.match(r'^frame_(\d+)(?:_jpg)?$', base)
    if m:
        return 'sequence_frames', int(m.group(1))

    # 6. Nenhum padrão reconhecido -> excluído do split (ver auditoria)
    return None


def collect_all_images(dataset_path):
    """
    Consolida todas as imagens reais do dataset em uma única lista ordenada
    de caminhos absolutos (string).

    Suporta dois layouts:
      - Legado (ex: dataset-overall): train/images/ + valid/images/
      - Bruto/plano (ex: ds-overall-raw): images/ + labels/ diretamente na
        raiz do dataset.
    """
    exts = ['*.jpg', '*.jpeg', '*.png']
    all_images = []

    legacy_dirs = [dataset_path / split / 'images' for split in ['train', 'valid']]
    if any(d.exists() for d in legacy_dirs):
        for images_dir in legacy_dirs:
            if not images_dir.exists():
                print(f"[!] Aviso: diretório não encontrado, ignorando: {images_dir}")
                continue
            for ext in exts:
                all_images.extend(images_dir.glob(ext))
    else:
        images_dir = dataset_path / 'images'
        if not images_dir.exists():
            print(f"[!] Erro: nenhum layout reconhecido em {dataset_path} "
                  f"(esperado train/images+valid/images ou images/)")
            return []
        for ext in exts:
            all_images.extend(images_dir.glob(ext))

    return sorted(str(p.absolute()) for p in all_images)


def create_temporal_folds(dataset_path, output_dir, purge_frames):
    print("=======================================================")
    print(" Geração de K-Fold Temporal - YOLOv8 Overall (Baseline)")
    print(f" Diretório do dataset: {dataset_path}")
    print(f" Diretório de saída  : {output_dir}")
    print(f" Purge buffer: {purge_frames} frames em cada lado do bloco de validação")
    print("=======================================================")

    all_images = collect_all_images(dataset_path)
    print(f"Total de imagens reais encontradas (train + valid): {len(all_images)}")

    if not all_images:
        print(f"[!] Erro: Nenhuma imagem encontrada em {dataset_path}")
        return

    # Agrupar por vídeo, descartando imagens sem padrão de nome reconhecido
    video_groups = defaultdict(list)
    excluded_images = []
    for img_path in all_images:
        parsed = parse_filename_info(img_path)
        if parsed is None:
            excluded_images.append(img_path)
            continue
        vid_id, frame_idx = parsed
        video_groups[vid_id].append((frame_idx, img_path))

    print(f"Imagens excluídas (nome fora do padrão, ordem cronológica não garantida): {len(excluded_images)}")
    print(f"Imagens válidas para o split temporal: {sum(len(v) for v in video_groups.values())}")
    print(f"Total de fontes/vídeos identificados: {len(video_groups)}")
    for vid_id, img_list in sorted(video_groups.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  - Vídeo '{vid_id}': {len(img_list)} frames")

    # Para cada vídeo, ordenar frames por frame_idx
    for vid_id in video_groups:
        video_groups[vid_id].sort(key=lambda x: x[0])

    # Dividir cada vídeo em 5 blocos temporais contínuos, com purge nas fronteiras
    folds_train = [[] for _ in range(NUM_FOLDS)]
    folds_val = [[] for _ in range(NUM_FOLDS)]
    total_purged = 0

    for vid_id, img_list in video_groups.items():
        n_frames = len(img_list)
        block_size = n_frames / NUM_FOLDS

        # Atribuir cada posição (índice cronológico dentro do vídeo) a um bloco (0 a 4)
        block_of_idx = [min(int(idx / block_size), NUM_FOLDS - 1) for idx in range(n_frames)]

        for f in range(NUM_FOLDS):
            # Fronteiras do bloco de validação do fold f dentro deste vídeo
            positions_in_block = [i for i, b in enumerate(block_of_idx) if b == f]
            if not positions_in_block:
                continue
            block_start = min(positions_in_block)
            block_end = max(positions_in_block) + 1  # exclusivo

            purge_start = max(0, block_start - purge_frames)
            purge_end = min(n_frames, block_end + purge_frames)

            for idx, (frame_idx, img_path) in enumerate(img_list):
                if block_of_idx[idx] == f:
                    folds_val[f].append(img_path)
                elif purge_start <= idx < purge_end:
                    # Dentro da zona de purga: nem treino, nem validação
                    total_purged += 1
                else:
                    folds_train[f].append(img_path)

    print(f"Total de amostras removidas pela zona de purga (todos os folds somados): {total_purged}")

    # Escrever arquivos train_fold_X.txt, val_fold_X.txt e data_fold_X.yaml
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(NUM_FOLDS):
        train_txt = output_dir / f"train_fold_{fold}.txt"
        val_txt = output_dir / f"val_fold_{fold}.txt"
        yaml_file = output_dir / f"data_fold_{fold}.yaml"

        train_txt.write_text('\n'.join(folds_train[fold]))
        val_txt.write_text('\n'.join(folds_val[fold]))

        yaml_content = f"""# YAML do Fold {fold} - Temporal Split com purge buffer ({purge_frames} frames)
train: {train_txt.absolute()}
val: {val_txt.absolute()}

nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""
        yaml_file.write_text(yaml_content)

        print(f"  [OK] Fold {fold}: {len(folds_train[fold])} treino, {len(folds_val[fold])} val | YAML: {yaml_file.name}")

    # Auditoria de Vazamento (sobreposição exata de arquivos)
    print("\n--- Auditoria de Vazamento de Dados ---")
    for fold in range(NUM_FOLDS):
        val_set = set(folds_val[fold])
        train_set = set(folds_train[fold])
        intersection = val_set.intersection(train_set)
        assert len(intersection) == 0, f"ERRO: Sobreposição detectada no Fold {fold}!"

    print("  >>> SUCESSO: 0% de sobreposição exata de arquivos entre Treino e Validação!")
    print("  >>> AVISO: purge buffer reduz o vazamento por adjacência temporal, mas NÃO elimina")
    print("             vazamento de cena/sessão quando poucas sessões de vídeo dominam o dataset.")

    # Relatório de auditoria para o TCC
    report_path = output_dir / "kfold_audit_report.txt"
    lines = [
        "Relatório de Auditoria - yolo-8-overall (Baseline Monolítico)",
        f"Diretório de origem: {dataset_path}",
        f"Total de imagens encontradas: {len(all_images)}",
        f"Imagens excluídas (fora do padrão regex, ordem cronológica não garantida): {len(excluded_images)}",
        f"Imagens válidas usadas no split: {sum(len(v) for v in video_groups.values())}",
        f"Vídeos/sessões identificados: {len(video_groups)}",
    ]
    for vid_id, img_list in sorted(video_groups.items(), key=lambda x: len(x[1]), reverse=True):
        lines.append(f"  - '{vid_id}': {len(img_list)} frames")
    lines.append(f"Purge buffer aplicado: {purge_frames} frames em cada lado do bloco de validação")
    lines.append(f"Total de amostras removidas pela zona de purga (todos os folds): {total_purged}")
    lines.append("")
    lines.append("Contagem por fold (após purge):")
    for fold in range(NUM_FOLDS):
        lines.append(f"  Fold {fold}: treino={len(folds_train[fold])}, val={len(folds_val[fold])}")
    lines.append("")
    lines.append(f"Imagens excluídas (lista completa, {len(excluded_images)} arquivos):")
    lines.extend(f"  {p}" for p in excluded_images)
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  [OK] Relatório de auditoria salvo em: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Geração de Folds Temporais para YOLOv8 Overall com purge buffer"
    )
    parser.add_argument(
        "--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH),
        help="Caminho base do dataset consolidado (contendo train/images e valid/images)"
    )
    parser.add_argument(
        "--purge-frames", type=int, default=DEFAULT_PURGE_FRAMES,
        help=(
            "Número de frames removidos de cada lado do bloco de validação "
            "(zona de purga/embargo) para reduzir vazamento por adjacência "
            f"temporal. Default: {DEFAULT_PURGE_FRAMES}. Ajuste conforme a "
            "taxa real de extração de frames do vídeo (frames muito "
            "próximos no tempo são quase-duplicatas visuais)."
        )
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Diretório onde os arquivos train/val/yaml/relatório serão escritos"
    )
    args = parser.parse_args()

    create_temporal_folds(
        Path(args.dataset_path),
        Path(args.output_dir),
        args.purge_frames,
    )

    print("\n=======================================================")
    print(" Processo concluído!")
    print("=======================================================")


if __name__ == '__main__':
    main()
