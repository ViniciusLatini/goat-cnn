#!/usr/bin/env python3
"""
check_session_leakage.py

Verifica se os vídeos de teste em ./testes pertencem a alguma sessão de
gravação usada no treinamento (folds K-Fold sobre ds-overall-raw).

Estratégia (não há metadados de data/hora nos arquivos .avi de teste):
  1. Extrai o conjunto de sessões de treino (Cam0X_YYYY_MM_DD_HH_MM_SS) a
     partir de train_fold_*.txt / val_fold_*.txt, e verifica se a DATA
     (DD-MM) presente no nome do vídeo de teste coincide com a data de
     alguma sessão de treino (sinal grosseiro, apenas informativo: câmera
     e sessão especifica podem ainda assim ser diferentes).
  2. Faz uma pré-seleção grosseira por average-hash (8x8, distância de
     Hamming) entre frames amostrados do vídeo de teste e imagens das
     sessões candidatas (ou de todas as sessões, se nenhuma data bater).
     Como as cenas do capril são visualmente muito parecidas entre si
     (mesmo fundo/iluminação), o aHash sozinho gera falsos positivos.
  3. Confirma os melhores candidatos do passo 2 com uma comparação
     pixel-a-pixel (MSE em 160x160) — um frame realmente duplicado
     (mesma captura, possivelmente recomprimida) tem MSE próximo de 0,
     bem abaixo do MSE típico entre dois frames DIFERENTES da MESMA
     sessão de treino (calibrado empiricamente em ~750-1200 neste
     dataset). Só é reportado como vazamento confirmado abaixo do limiar
     MSE_DUPLICATE_THRESHOLD.
  4. Classifica cada vídeo como "SEM EVIDÊNCIA DE VAZAMENTO" ou
     "VAZAMENTO CONFIRMADO", e reporta separadamente a coincidência de
     data (apenas informativa).

Uso:
    python check_session_leakage.py [--report caminho/relatorio.txt]
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATASET_IMAGES_DIR = Path(
    "/run/media/vini/061a1dd7-8e19-49bb-a45c-87c4603733ea/vini/ds-overall-raw/images"
)
FOLD_LIST_FILES = list(BASE_DIR.glob("train_fold_*.txt")) + list(BASE_DIR.glob("val_fold_*.txt"))
TESTES_DIR = BASE_DIR / "testes"

SESSION_RE = re.compile(r"(Cam\d+_(\d{4})_(\d{2})_(\d{2})_\d{2}_\d{2}_\d{2})")
VIDEO_DATE_RE = re.compile(r"(\d{2})-(\d{2})")

HASH_SIZE = 8
FRAMES_PER_VIDEO = 12
IMAGES_PER_SESSION_SAMPLE = 40
HAMMING_PREFILTER_THRESHOLD = 12  # grosseiro: só decide quais pares vão para verificação por MSE
TOP_K_CANDIDATES = 8              # nº de pares (frame vídeo, imagem treino) verificados por MSE
MSE_COMPARE_SIZE = 160
MSE_DUPLICATE_THRESHOLD = 60      # abaixo disso = mesma captura (compressão/reencode); acima = cena apenas parecida


def load_training_sessions():
    sessions = {}
    for list_file in FOLD_LIST_FILES:
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                m = SESSION_RE.search(line)
                if not m:
                    continue
                full, year, month, day = m.groups()
                sessions.setdefault(full, {"year": year, "month": month, "day": day, "images": []})
    return sessions


def index_session_images(sessions):
    if not DATASET_IMAGES_DIR.exists():
        print(f"[AVISO] Pasta de imagens do dataset não encontrada: {DATASET_IMAGES_DIR}")
        return
    for img_path in DATASET_IMAGES_DIR.iterdir():
        m = SESSION_RE.search(img_path.name)
        if not m:
            continue
        session_key = m.group(1)
        if session_key in sessions:
            sessions[session_key]["images"].append(img_path)


def average_hash(img_gray):
    small = cv2.resize(img_gray, (HASH_SIZE, HASH_SIZE), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = (small > mean).flatten()
    return np.packbits(bits)


def hamming_distance(h1, h2):
    return int(np.unpackbits(np.bitwise_xor(h1, h2)).sum())


def sample_video_frames(video_path, n_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total <= 0:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    indices = np.linspace(0, total - 1, num=min(n_frames, total), dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)
    cap.release()
    return frames


def sample_session_images(image_paths, n_sample=IMAGES_PER_SESSION_SAMPLE):
    if len(image_paths) > n_sample:
        step = len(image_paths) / n_sample
        return [image_paths[int(i * step)] for i in range(n_sample)]
    return list(image_paths)


def mse_at(img_a, img_b, size=MSE_COMPARE_SIZE):
    a = cv2.resize(img_a, (size, size)).astype(np.float64)
    b = cv2.resize(img_b, (size, size)).astype(np.float64)
    return float(np.mean((a - b) ** 2))


def find_date_matched_sessions(video_name, sessions):
    """Apenas informativo: retorna sessões cuja data de calendário coincide com a do nome do vídeo."""
    m = VIDEO_DATE_RE.search(video_name)
    if not m:
        return [], None
    a, b = m.groups()
    matched = [
        key for key, info in sessions.items()
        if (info["month"], info["day"]) in ((a, b), (b, a))
    ]
    return matched, (a, b)


def check_video(video_path, sessions):
    date_matched_sessions, parsed_date = find_date_matched_sessions(video_path.name, sessions)
    # Universo de comparação: sessões com a mesma data (se houver) ou todas, como fallback exaustivo.
    candidate_keys = date_matched_sessions if date_matched_sessions else list(sessions.keys())

    video_frames = sample_video_frames(video_path)
    video_frame_hashes = [(f, average_hash(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))) for f in video_frames]

    # Estágio 1: pré-filtro grosseiro por aHash (rápido, gera candidatos, não decide sozinho).
    hash_candidates = []  # (distance, video_frame, img_path)
    for key in candidate_keys:
        info = sessions[key]
        if not info["images"]:
            continue
        for img_path in sample_session_images(info["images"]):
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue
            img_hash = average_hash(img_gray)
            for vframe, vhash in video_frame_hashes:
                d = hamming_distance(vhash, img_hash)
                if d <= HAMMING_PREFILTER_THRESHOLD:
                    hash_candidates.append((d, key, vframe, img_path))

    hash_candidates.sort(key=lambda x: x[0])
    top_candidates = hash_candidates[:TOP_K_CANDIDATES]

    # Estágio 2: confirmação pixel-a-pixel (MSE) apenas nos melhores candidatos do pré-filtro.
    best_mse = None
    best_match = None
    for _, key, vframe, img_path in top_candidates:
        img_color = cv2.imread(str(img_path))
        if img_color is None:
            continue
        m = mse_at(vframe, img_color)
        if best_mse is None or m < best_mse:
            best_mse = m
            best_match = (key, img_path)

    is_leak = best_mse is not None and best_mse <= MSE_DUPLICATE_THRESHOLD
    return {
        "video": video_path.name,
        "parsed_date": parsed_date,
        "date_matches_training": bool(date_matched_sessions),
        "n_candidate_sessions": len(candidate_keys),
        "n_frames_sampled": len(video_frames),
        "n_hash_prefilter_hits": len(hash_candidates),
        "best_mse": best_mse,
        "best_match_session": best_match[0] if best_match else None,
        "best_match_image": best_match[1].name if best_match else None,
        "is_leak": is_leak,
    }


def main():
    parser = argparse.ArgumentParser(description="Verifica vazamento de sessões de treino nos vídeos de teste.")
    parser.add_argument("--report", type=str, default=None, help="Caminho do arquivo de relatório txt (opcional).")
    args = parser.parse_args()

    print("=" * 70)
    print(" VERIFICAÇÃO DE VAZAMENTO DE SESSÃO (VÍDEOS DE TESTE x TREINO)")
    print("=" * 70)

    sessions = load_training_sessions()
    print(f"Sessões de treino encontradas nos folds: {len(sessions)}")
    index_session_images(sessions)
    total_imgs = sum(len(s["images"]) for s in sessions.values())
    print(f"Imagens de treino indexadas: {total_imgs}")

    videos = sorted(TESTES_DIR.glob("*.avi")) + sorted(TESTES_DIR.glob("*.mp4"))
    if not videos:
        print(f"[AVISO] Nenhum vídeo encontrado em {TESTES_DIR}")
        return []

    results = []
    for v in videos:
        print(f"\n>>> Analisando: {v.name}")
        res = check_video(v, sessions)
        results.append(res)
        status = "VAZAMENTO CONFIRMADO" if res["is_leak"] else "SEM EVIDÊNCIA DE VAZAMENTO"
        print(f"  Data extraída do nome: {res['parsed_date']} (coincide com data de treino: {'sim' if res['date_matches_training'] else 'não'})")
        print(f"  Sessões comparadas: {res['n_candidate_sessions']}")
        print(f"  Frames amostrados do vídeo: {res['n_frames_sampled']}")
        print(f"  Candidatos após pré-filtro (aHash): {res['n_hash_prefilter_hits']}")
        print(f"  Menor MSE (pixel, 160x160) encontrado: {res['best_mse']}")
        if res["best_match_session"]:
            print(f"  Imagem de treino mais próxima: {res['best_match_session']} / {res['best_match_image']}")
        print(f"  >> Status: {status}")

    print("\n" + "=" * 70)
    print(" RESUMO")
    print("=" * 70)
    for r in results:
        status = "VAZAMENTO CONFIRMADO" if r["is_leak"] else "OK (sem evidência)"
        date_flag = " [data coincide c/ treino]" if r["date_matches_training"] else ""
        print(f"  {r['video']:<30} -> {status} (MSE mínimo = {r['best_mse']}){date_flag}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(" VERIFICAÇÃO DE VAZAMENTO DE SESSÃO (VÍDEOS DE TESTE x TREINO)\n")
            f.write("=" * 70 + "\n")
            f.write(f"Sessões de treino: {len(sessions)} | Imagens indexadas: {total_imgs}\n")
            f.write(f"Pré-filtro aHash {HASH_SIZE}x{HASH_SIZE} (limiar Hamming <= {HAMMING_PREFILTER_THRESHOLD}), "
                    f"confirmação por MSE pixel-a-pixel (limiar <= {MSE_DUPLICATE_THRESHOLD} = duplicata confirmada)\n")
            f.write("Referência de calibração: MSE self-vs-self = 0; MSE entre dois frames DIFERENTES "
                    "da MESMA sessão de treino ~750-1200 neste dataset.\n")
            f.write("-" * 70 + "\n")
            for r in results:
                status = "VAZAMENTO CONFIRMADO" if r["is_leak"] else "SEM EVIDÊNCIA DE VAZAMENTO"
                f.write(f"\n* Vídeo: {r['video']}\n")
                f.write(f"  - Data extraída do nome: {r['parsed_date']}\n")
                f.write(f"  - Data coincide com alguma sessão de treino: {'sim' if r['date_matches_training'] else 'não'}\n")
                f.write(f"  - Sessões comparadas: {r['n_candidate_sessions']}\n")
                f.write(f"  - Frames amostrados: {r['n_frames_sampled']}\n")
                f.write(f"  - Candidatos após pré-filtro (aHash): {r['n_hash_prefilter_hits']}\n")
                f.write(f"  - Menor MSE (pixel) encontrado: {r['best_mse']}\n")
                f.write(f"  - Imagem de treino mais próxima: {r['best_match_session']} / {r['best_match_image']}\n")
                f.write(f"  - Status: {status}\n")
            f.write("\n" + "=" * 70 + "\n")
        print(f"\nRelatório de vazamento salvo em: {report_path.resolve()}")

    return results


if __name__ == "__main__":
    main()
