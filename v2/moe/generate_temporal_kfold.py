#!/usr/bin/env python3
"""
generate_temporal_kfold.py

Script para gerar divisões de Validação Cruzada K-Fold (5 folds) para os modelos
especialistas YOLO, lendo diretamente do diretório montado:
/run/media/vini/061a1dd7-8e19-49bb-a45c-87c4603733ea/vini/ds

Estratégia:
1. Agrupa imagens por vídeo/sessão de gravação (via parsing do nome do arquivo).
2. Descarta imagens cujo nome não bate com nenhum padrão de vídeo/frame conhecido
   (não é possível garantir sua posição cronológica real, então não podem ser
   posicionadas com segurança em um bloco temporal contíguo).
3. Ordena os frames cronologicamente por índice de frame dentro de cada vídeo.
4. Aplica Bloqueio Temporal Contínuo (Contiguous Time-Block Split / Temporal K-Fold):
   - Cada fold recebe uma janela temporal contínua de 20% do vídeo para validação.
   - Os outros 80% (4 blocos restantes) são usados no treino.
5. Aplica uma zona de purga (purge/embargo buffer) de N frames em ambos os lados
   de cada bloco de validação: frames dentro dessa janela são descartados do fold
   (não entram nem em treino nem em validação), pois são temporalmente próximos
   o suficiente do bloco de validação para serem quase-duplicatas visuais dele.
   Isso mitiga (mas não elimina — ver CLAUDE.md/discussão) o vazamento por
   adjacência temporal entre treino e validação.
6. Garante 0% de sobreposição *exata* de arquivos entre treino e validação.

Observação importante: o purge buffer reduz o vazamento por adjacência temporal
(frames quase-idênticos na fronteira do bloco), mas NÃO elimina o vazamento de
cena/sessão (mesmo fundo, mesmos indivíduos, mesma câmera) quando um especialista
tem poucos vídeos-fonte independentes. Ver auditoria correspondente para detalhes.
"""

import re
import argparse
from pathlib import Path
from collections import defaultdict

NUM_FOLDS = 5
DEFAULT_PURGE_FRAMES = 15
DEFAULT_DATASET_BASE = Path('/run/media/vini/061a1dd7-8e19-49bb-a45c-87c4603733ea/vini/ds')

SPECIALIST_MAPPING = {
    'yolo-8-front-dia': 'ds_front_dia',
    'yolo-8-front-noite': 'ds_front_noite',
    'yolo-8-sup-dia': 'ds_sup_dia',
    'yolo-8-sup-noite': 'ds_sup_noite'
}

def parse_filename_info(filepath_str):
    """
    Extrai o ID do vídeo/sessão e o índice do frame a partir do nome da imagem.
    Retorna None quando o nome não bate com nenhum padrão conhecido, pois nesse
    caso não há garantia de que a imagem possa ser posicionada corretamente na
    sequência cronológica do vídeo.
    """
    name = Path(filepath_str).name
    # Remove hash do roboflow (.rf.xxxx.jpg)
    base = name.split('.rf.')[0]

    # 1. Padrão: <video_prefix>_frame_<frame_num>_jpg
    m = re.match(r'^(.*?)_frame_(\d+)_jpg$', base)
    if m:
        return m.group(1), int(m.group(2))

    # 2. Padrão: <video_prefix>-<frame_num>_jpg
    m = re.match(r'^(.*?)-(\d+)_jpg$', base)
    if m:
        return m.group(1), int(m.group(2))

    # 3. Padrão: <video_prefix>_random_<frame_num>_jpg
    m = re.match(r'^(.*?)_random_(\d+)_jpg$', base)
    if m:
        return m.group(1), int(m.group(2))

    # 4. Padrão: frame_(\d+)_jpg (sequência sem prefixo de câmera/sessão)
    m = re.match(r'^frame_(\d+)_jpg$', base)
    if m:
        return 'sequence_frames', int(m.group(1))

    # 5. Padrão: <video_prefix>-frame_<frame_num>.<ext> (extração direta de frames,
    #    sem passar pelo export do Roboflow, portanto sem sufixo ".rf.<hash>" e com
    #    extensão real preservada em vez de mangled "_jpg")
    m = re.match(r'^(.*?)-frame_(\d+)\.(?:jpg|jpeg|png)$', base, re.IGNORECASE)
    if m:
        return m.group(1), int(m.group(2))

    # 6. Nenhum padrão reconhecido -> excluído do split (ver auditoria)
    return None

def create_temporal_folds(specialist_dir, ds_folder, dataset_base_path, purge_frames):
    specialist_path = Path('/home/vini/tcc/moe') / specialist_dir
    images_dir = dataset_base_path / ds_folder / 'images'

    print(f"\n=======================================================")
    print(f" Processando Especialista: {specialist_dir}")
    print(f" Diretório das Imagens: {images_dir}")
    print(f" Purge buffer: {purge_frames} frames em cada lado do bloco de validação")
    print(f"=======================================================")

    if not images_dir.exists():
        print(f"[!] Erro: Diretório de imagens não encontrado: {images_dir}")
        return

    # Buscar todas as imagens no diretório real (recursivo: alguns exports
    # deixam as imagens novas em uma subpasta aninhada, ex. images/images/)
    exts = ['*.jpg', '*.png', '*.jpeg']
    all_images = []
    for ext in exts:
        all_images.extend(list(images_dir.rglob(ext)))

    all_images = sorted([str(p.absolute()) for p in all_images])
    print(f"Total de imagens reais encontradas: {len(all_images)}")

    if not all_images:
        print(f"[!] Erro: Nenhuma imagem encontrada em {images_dir}")
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
    for fold in range(NUM_FOLDS):
        train_txt = specialist_path / f"train_fold_{fold}.txt"
        val_txt = specialist_path / f"val_fold_{fold}.txt"
        yaml_file = specialist_path / f"data_fold_{fold}.yaml"

        train_txt.write_text('\n'.join(folds_train[fold]))
        val_txt.write_text('\n'.join(folds_val[fold]))

        yaml_content = f"""# YAML do Fold {fold} - Temporal Split com purge buffer ({purge_frames} frames)
train: {train_txt.absolute()}
val: {val_txt.absolute()}

nc: 1
names: ['goat']
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

    print(f"  >>> SUCESSO: 0% de sobreposição exata de arquivos entre Treino e Validação!")
    print(f"  >>> AVISO: purge buffer reduz o vazamento por adjacência temporal, mas NÃO elimina")
    print(f"             vazamento de cena/sessão em especialistas com poucos vídeos-fonte.")

    # Relatório de auditoria para o TCC
    report_path = specialist_path / "kfold_audit_report.txt"
    lines = [
        f"Relatório de Auditoria - {specialist_dir}",
        f"Diretório de origem: {images_dir}",
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
    parser = argparse.ArgumentParser(description="Geração de Folds Temporais para YOLOv8 com purge buffer")
    parser.add_argument("--dataset-base-path", type=str, default=str(DEFAULT_DATASET_BASE),
                        help="Caminho base onde as pastas de dataset estão armazenadas")
    parser.add_argument("--purge-frames", type=int, default=DEFAULT_PURGE_FRAMES,
                        help=("Número de frames removidos de cada lado do bloco de validação "
                              "(zona de purga/embargo) para reduzir vazamento por adjacência "
                              f"temporal. Default: {DEFAULT_PURGE_FRAMES}. Ajuste conforme a taxa "
                              "real de extração de frames do vídeo (frames muito próximos no tempo "
                              "são quase-duplicatas visuais)."))
    args = parser.parse_args()

    ds_base = Path(args.dataset_base_path)

    for spec_dir, ds_folder in SPECIALIST_MAPPING.items():
        create_temporal_folds(spec_dir, ds_folder, ds_base, args.purge_frames)

    print("\n=======================================================")
    print(" Processo concluído em todos os especialistas!")
    print("=======================================================")

if __name__ == '__main__':
    main()
