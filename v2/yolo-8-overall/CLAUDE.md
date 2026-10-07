# CLAUDE.md - Contexto Geral, Arquitetura e Organização do Projeto

Este arquivo concentra o contexto completo, histórico de desenvolvimento, arquitetura técnica e organização do projeto **YOLOv8 Overall (Modelo Monolítico Generalista)** no contexto do **TCC: Detecção de Caprinos via Mixture of Experts (MoE) vs. Baseline Generalista**.

---

## 1. Visão Geral do Projeto

* **Tema:** Detecção automatizada de caprinos em instalações de confinamento (capril) sob variações severas de iluminação (dia vs. noite/infravermelho) e posicionamento de câmera (frontal vs. superior/zenital).
* **Papel do Módulo Overall:** Modelo **Monolítico Generalista (Baseline Global)** treinado sobre a totalidade unificada dos dados de todos os cenários.
* **Objetivo Científico no TCC:** Servir como *benchmark* (linha de base comparativa) para avaliar se a arquitetura **Mixture of Experts (MoE)** — composta por Gating Network (EfficientNet-B0) + 4 especialistas YOLOv8m — apresenta ganhos de acurácia, robustez e generalização frente a uma abordagem monolítica tradicional que aprende todos os cenários conjuntamente.
* **Classe Detectada:** `goat` (`nc: 1`).

---

## 2. Dados e Dataset

* **Caminho Base do Dataset (atual, usado nos folds vigentes):**
  `/run/media/vini/061a1dd7-8e19-49bb-a45c-87c4603733ea/vini/ds-overall-raw/`
* **Subdiretórios de Origem (layout plano, sem split train/valid prévio):**
  * `images/` e `labels/`
* **Volume de Dados:** 8.604 imagens anotadas cobrindo múltiplos ângulos, horários e câmeras (`Cam01`–`Cam04`), agrupadas em 36 vídeos/sessões distintos.
* **Padrões de Nomenclatura das Imagens:**
  * Exportado via Roboflow: `Cam0X_YYYY_MM_DD_HH_MM_SS_capril[_mp4|_avi]-XXXX_jpg.rf.HASH.jpg`
  * Frame bruto extraído diretamente do vídeo (sem hash Roboflow): `Cam0X_YYYY_MM_DD_HH_MM_SS_capril-frame_XXXX.jpg`
* **Configurações YAML:** `data_fold_0.yaml` a `data_fold_4.yaml` mapeando as listas de treino (`train_fold_X.txt`) e validação (`val_fold_X.txt`), gerados por `generate_temporal_kfold.py` (purge buffer de 15 frames, 0% de sobreposição exata treino/val — ver `kfold_audit_report.txt`).
* **Dataset legado (usado apenas nos resultados históricos da Seção 4.2):** `dataset-overall/` (~3.755 imagens, layout `train/images`+`valid/images`). Os folds antigos gerados sobre ele foram preservados em `backup_dataset_overall_folds_old/` antes da regeneração para o novo dataset bruto.
* **Status atual (2026-09-03):** o K-Fold sobre `ds-overall-raw/` (8.604 imagens) foi **treinado por completo** — ver Seção 4.1 para os resultados atuais e Seção 7 para os testes de inferência em vídeo realizados sobre o modelo resultante.

---

## 3. Arquitetura e Treinamento

### 3.1 Modelo e Hiperparâmetros
* **Backbone / Arquitetura:** YOLOv8 Médio (`yolov8m.pt`).
* **Resolução de Entrada (`imgsz`):** $640 \times 640$.
* **Épocas:** 300 épocas máximas.
* **Early Stopping (`patience`):** 30 épocas com base no mAP50.
* **Batch Size:** AutoBatch (`batch=-1`), ajustando dinamicamente ao espaço de VRAM da GPU.
* **Dataloading / Cache:** `cache='ram'` para mitigar o gargalo de leitura em HDs mecânicos, com `workers=8`.
* **Dispositivo:** CUDA (`cuda:0` / NVIDIA GeForce RTX 5070 com 12GB VRAM).

### 3.2 Estratégia de Validação Cruzada (5-Fold)
* **Objetivo:** Avaliar a estabilidade do modelo e garantir que as métricas reportadas no TCC não sejam fruto de um *split* favorável isolado.
* **Prevenção de Vazamento Temporal (*Data Leakage*):**
  * *Contiguous Time-Block Split / Temporal K-Fold*: Imagens provenientes da mesma sequência de vídeo/sessão são agrupadas cronologicamente. Cada fold retém uma janela temporal contígua de 20% para validação e 80% para treino.
  * *Purge/Embargo Buffer*: Remoção de uma margem de 15 frames em cada borda do bloco de validação para eliminar quase-duplicatas visuais na fronteira temporal.
  * Auditoria com 0% de sobreposição exata de arquivos entre treino e teste.
* **Auditoria real do split vigente (`kfold_audit_report.txt`, sobre `ds-overall-raw/`):**
  * Total de imagens encontradas/válidas: 8.604 (0 excluídas por fugirem do padrão de nomenclatura).
  * Vídeos/sessões identificados: 36 (variando de 40 a 1.670 frames por sessão).
  * Amostras removidas pela zona de purga (soma de todos os folds): 4.234.
  * Contagem por fold após purge — Fold 0: treino 6.340 / val 1.724; Fold 1: treino 5.847 / val 1.720; Fold 2: treino 5.803 / val 1.721; Fold 3: treino 5.847 / val 1.720; Fold 4: treino 6.345 / val 1.719.

### 3.3 Execução do Treinamento (sessão de 2026-09-03)
* **Script utilizado:** `run_kfold_trainings.py` (substitui o `k-fold-train.py` legado para este dataset) — treina os 5 folds **sequencialmente** na mesma GPU (RTX 5070, 12GB), pois rodar folds em paralelo nesta máquina causaria *CUDA Out of Memory*. Antes de iniciar, o script limpa caches `.cache` antigos (do split aleatório anterior) para evitar `_pickle.UnpicklingError`.
* Cada fold parte do zero a partir dos pesos pré-treinados `yolov8m.pt` (COCO) e treina com `patience=30` (parada antecipada por estagnação do mAP50), `imgsz=640`, `batch=-1` (AutoBatch) e `cache='ram'`.
* **Épocas efetivamente treinadas por fold** (limite de 300, mas todos pararam antes por *early stopping*): Fold 1 = 224 · Fold 2 = 126 · Fold 3 = 106 · Fold 4 = 139 · Fold 5 = 145.
* **Tempo de treinamento por fold** (campo `time` do `results.csv`, GPU RTX 5070): Fold 1 ≈ 4h39min · Fold 2 ≈ 2h29min · Fold 3 ≈ 2h05min · Fold 4 ≈ 2h45min · Fold 5 ≈ 3h05min — **tempo total da rodada completa ≈ 15h e 3min**, executados em sequência entre 2026-09-03 ~02:44h e ~17:49h.
* Ao final, `metric-k-fold.py` foi executado para ler o `results.csv` de cada fold, extrair a época de pico de mAP50 de cada um, eleger o vencedor pelo maior mAP50 e copiar automaticamente `weights/best.pt` desse fold para a raiz do projeto (`best.pt`). O vencedor desta rodada foi o **Fold 3** — ver Seção 4.1.

---

## 4. Resultados Consolidados dos Treinamentos (K-Fold Overall)

### 4.1 Resultados Atuais — dataset `ds-overall-raw/` (8.604 imagens, 36 sessões)

Rodada completa treinada e concluída em 2026-09-03 (ver Seção 3.3 para o processo de execução). Métricas do ponto de melhor mAP50 de cada fold, extraídas de `runs/detect/runs/detect/fold_X/results.csv` pelo script `metric-k-fold.py`:

| Fold | Época de Pico | Precisão (P) | Revocação (R) | F1-Score | mAP50 | mAP50-95 | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 96 | 0.9914 | 0.9938 | 0.9926 | 0.9944 | 0.8700 | Estável |
| **Fold 2** | 60 | 0.9945 | 0.9942 | 0.9943 | 0.9948 | 0.8707 | Estável |
| **Fold 3** | 54 | 0.9977 | 0.9986 | 0.9982 | 0.9950 | 0.8965 | **Melhor Geral** |
| **Fold 4** | 33 | 0.9967 | 0.9971 | 0.9969 | 0.9950 | 0.8839 | Estável |
| **Fold 5** | 25 | 0.9939 | 0.9948 | 0.9943 | 0.9949 | 0.8653 | Estável |
| **Média** | — | **0.9948** | **0.9957** | **0.9953** | **0.9948** | **0.8773** | **Alta Convergência** |

* **Pesos selecionados:** o **Fold 3** venceu por maior mAP50 (0.9950, com o melhor mAP50-95 do grupo, 0.8965, e o melhor F1 geral, 0.9982). Seus pesos (`runs/detect/runs/detect/fold_3/weights/best.pt`) foram copiados automaticamente por `metric-k-fold.py` para a raiz do projeto como `best.pt` — **este é o `best.pt` atualmente em uso**, e foi o modelo usado nos testes de inferência em vídeo da Seção 7.
* **Leitura dos resultados:** mAP50 é praticamente idêntico entre os 5 folds (0.9944–0.9950), confirmando alta estabilidade do split temporal — a métrica não depende de qual janela de tempo caiu em validação. O mAP50-95 (mais rigoroso, exige sobreposição de caixa mais precisa) tem variação um pouco maior entre folds (0.8653–0.8965), o que é esperado dado o dataset bruto (frames extraídos diretamente de vídeo, sem a curadoria/qualidade de anotação do dataset legado da Seção 4.2).
* Comparado ao dataset legado (Seção 4.2), este dataset é ~2,3× maior (8.604 vs. 3.755 imagens) e usa frames brutos extraídos por sessão de vídeo (ao invés de um export já processado pelo Roboflow), o que explica o mAP50-95 médio um pouco mais baixo (0.8773 vs. 0.9085) apesar do mAP50 permanecer no mesmo patamar (~0.995).

### 4.2 Resultados Históricos (referência) — dataset legado `dataset-overall/` (~3.755 imagens)

> Mantido apenas como referência histórica do TCC; **não reflete mais o `best.pt` atual** (substituído pelos resultados da Seção 4.1).

| Fold | Época de Pico | Precisão (P) | Revocação (R) | F1-Score | mAP50 | mAP50-95 | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 107 | 0.9992 | 0.9970 | 0.9981 | 0.9950 | 0.9213 | Estável |
| **Fold 2** | 77 | 0.9970 | 0.9982 | 0.9976 | 0.9949 | 0.9146 | Estável |
| **Fold 3** | 19 | 0.9971 | 0.9973 | 0.9972 | 0.9950 | 0.8959 | Estável |
| **Fold 4** | 50 | 0.9975 | 0.9974 | 0.9974 | 0.9950 | 0.9110 | Estável |
| **Fold 5** | 24 | 0.9987 | 0.9992 | 0.9989 | 0.9950 | 0.8999 | Melhor Geral (na época) |
| **Média** | — | **0.9979** | **0.9978** | **0.9978** | **0.9950** | **0.9085** | **Alta Convergência** |

* Os folds gerados sobre este dataset foram preservados em `backup_dataset_overall_folds_old/` antes da regeneração do K-Fold sobre `ds-overall-raw/`.

---

## 5. Estrutura e Organização do Diretório

```text
/home/vini/tcc/yolo-8-overall/
├── CLAUDE.md                         # [ESTE ARQUIVO] Documentação técnica completa do projeto overall
├── yolov8m.pt                        # Pesos pré-treinados oficiais YOLOv8m (Ultralytics)
├── yolo26n.pt                        # Pesos auxiliares (YOLO Nano)
├── best.pt                           # Melhor modelo consolidado selecionado do K-Fold
│
├── data_fold_0.yaml ... 4.yaml       # Arquivos YAML de configuração para cada um dos 5 folds
├── train_fold_0.txt ... 4.txt        # Listas de imagens pertencentes ao conjunto de treino por fold
├── val_fold_0.txt ... 4.txt          # Listas de imagens pertencentes ao conjunto de validação por fold
│
├── index.py                          # Script de treino padrão de fold único e teste rápido de predição
├── generate_temporal_kfold.py        # Gera os splits temporais (data_fold_X.yaml, train/val_fold_X.txt) com purge buffer
├── run_kfold_trainings.py            # Script atual de execução do treinamento K-Fold sequencial (5 folds, ds-overall-raw)
├── k-fold-train.py                   # Script de treinamento K-Fold legado (usado no dataset-overall antigo)
├── metric-k-fold.py                  # Análise e recuperação do melhor fold com cópia para best.pt
├── results.py                        # Geração de relatório textual e gráficos de loss/mAP (analysis_plots.png)
├── test_videos.py                    # Script de inferência em lote para vídeos com relatório automático (legado)
├── run_video_tests.py                # Orquestrador: checagem de vazamento de sessão + inferência YOLO, saída em pasta com timestamp
├── check_session_leakage.py          # Verifica se vídeos de ./testes são duplicatas de frames usados no treino (aHash + MSE pixel-a-pixel)
├── create_test_frames.py             # Utilitário para extração do primeiro frame de vídeos de teste
├── teste.py                          # Verificador de compatibilidade de ambiente CUDA / PyTorch GPU
│
├── testes/                           # Vídeos de teste para validação de inferência
│   ├── front-dia-04-05.avi
│   ├── front-noite-26-04.avi
│   ├── sup-dia-29-04.avi
│   └── sup-noite-29-04.avi
│
├── output/                           # Saídas de inferência dos vídeos
│   └── test_run_YYYYMMDD_HHMMSS/      # Uma pasta por execução de run_video_tests.py
│       ├── leakage_report.txt        # Resultado da checagem de vazamento de sessão (teste x treino)
│       ├── relatorio_teste.txt       # Relatório com FPS, tempo, caixas detectadas por vídeo
│       ├── *.avi (anotados)
│       └── labels/                   # Coordenadas das caixas preditas frame a frame (.txt)
│
└── runs/detect/                      # Histórico de treinos, checkpoints e logs do Ultralytics
    └── runs/detect/
        ├── fold_1/ (weights/best.pt, results.csv, confusion_matrix.png, ...)
        ├── fold_2/
        ├── fold_3/
        ├── fold_4/
        └── fold_5/
```

---

## 6. Ambiente e Execução

### 6.1 Ambiente Conda
* **Nome do Ambiente:** `tcc`
* **Interpretador Python:** `/home/vini/miniconda3/envs/tcc/bin/python`
* **Ativação:**
  ```bash
  conda activate tcc
  ```
* **Bibliotecas Principais:** `torch`, `torchvision`, `ultralytics` (YOLOv8), `opencv-python`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`.

---

### 6.2 Comandos Principais

#### 1. Verificar GPU e CUDA
```bash
cd /home/vini/tcc/yolo-8-overall
python teste.py
```

#### 2. Executar Treinamento K-Fold
```bash
cd /home/vini/tcc/yolo-8-overall
python run_kfold_trainings.py
```
*Treina os 5 folds sequencialmente (não paralelo — evita CUDA OOM em GPU única) a partir de `yolov8m.pt`, salvando cada fold em `runs/detect/runs/detect/fold_X/`. Aceita `--folds 0 2 4` para treinar só alguns, e `--epochs`/`--batch`/`--imgsz`/`--workers` para sobrescrever os padrões. `k-fold-train.py` é a versão legada usada no dataset antigo.*

#### 3. Recuperar o Melhor Modelo do K-Fold e Salvar em `best.pt`
```bash
cd /home/vini/tcc/yolo-8-overall
python metric-k-fold.py
```

#### 4. Gerar Relatório e Gráficos de Treinamento
```bash
cd /home/vini/tcc/yolo-8-overall
python results.py
```
*Gera `training_report.txt` e `analysis_plots.png`. **Atenção:** o script está com o caminho fixo em `./runs/detect/train/results.csv` (formato de treino de fold único), incompatível com a estrutura atual `runs/detect/runs/detect/fold_X/results.csv` do K-Fold — não foi executado na rodada atual (Seção 4.1); os números de lá foram extraídos diretamente pelo `metric-k-fold.py`. Para reaproveitar `results.py`, ajuste `RESULTS_CSV` para apontar a um `fold_X/results.csv` específico antes de rodar.*

#### 5. Testar Inferência em Lote nos Vídeos (com checagem de vazamento de sessão)
```bash
cd /home/vini/tcc/yolo-8-overall
python run_video_tests.py --model best.pt --source ./testes --conf 0.7 --iou 0.7
```
*Cria `output/test_run_<timestamp>/` contendo: `leakage_report.txt` (checa se algum vídeo de `./testes` é
duplicata de frames usados no treino — pré-filtro por average-hash + confirmação por MSE pixel-a-pixel
contra as sessões do K-Fold), os vídeos anotados, os `.txt` de caixas (`labels/`) e `relatorio_teste.txt`
(FPS, tempo, detecções por vídeo). Use `--skip-leakage-check` para pular a Etapa 1.*

*Script legado equivalente (sem checagem de vazamento, saída direta em `./output/`):*
```bash
python test_videos.py --model best.pt --source ./testes --output ./output --conf 0.7 --iou 0.7
```

#### 6. Extrair Primeiros Frames de Vídeos
```bash
cd /home/vini/tcc/yolo-8-overall
python create_test_frames.py
```

---

## 7. Testes de Inferência em Vídeo e Verificação de Vazamento de Sessão (2026-09-03)

Após a conclusão da rodada de K-Fold da Seção 4.1 e a atualização de `best.pt` (Fold 3), o modelo foi validado qualitativamente contra os 4 vídeos de `testes/` (clipes de 10s cada, um por combinação câmera×iluminação: `front-dia-04-05.avi`, `front-noite-26-04.avi`, `sup-dia-29-04.avi`, `sup-noite-29-04.avi`). Antes de reportar esses vídeos como um teste de generalização válido, era preciso confirmar que nenhum deles é, na verdade, um recorte de uma sessão já usada no treinamento — o que inflaria artificialmente os resultados.

### 7.1 Verificação de Vazamento de Sessão (`check_session_leakage.py`)

Os arquivos `.avi` de teste não carregam metadados de data/hora (`ffprobe` não retornou `creation_time`), e seus nomes não seguem o padrão `Cam0X_YYYY_MM_DD_HH_MM_SS` usado no dataset — então a checagem não podia ser feita apenas por nome de arquivo. Metodologia adotada, em duas etapas:

1. **Sinal de data (informativo):** extrai a data (`DD-MM`) do nome do vídeo de teste e verifica se coincide com a data de alguma das 34 sessões de treino identificadas em `train_fold_*.txt`/`val_fold_*.txt`. Por si só **não prova nem descarta** vazamento — várias sessões/câmeras diferentes podem compartilhar a mesma data.
2. **Confirmação visual (decisiva):** amostra 12 frames do vídeo de teste e compara contra as imagens de treino das sessões candidatas (mesma data, ou todas as 34 sessões se nenhuma data bater) em duas etapas:
   * *Pré-filtro grosseiro* por average-hash (8×8, distância de Hamming ≤ 12) — rápido, mas gera falsos positivos porque as cenas do capril são visualmente muito parecidas entre si (mesmo fundo, iluminação semelhante).
   * *Confirmação por MSE pixel-a-pixel* (imagens redimensionadas para 160×160) nos melhores candidatos do pré-filtro. Calibração empírica feita neste próprio dataset: MSE de uma imagem contra ela mesma = 0; MSE entre dois frames **diferentes de uma mesma sessão de treino** ≈ 750–1200 (o gado se move bastante entre frames extraídos). Um frame de teste só é reportado como **vazamento confirmado** se o MSE mínimo encontrado for ≤ 60 (ordem de grandeza de recompressão/reencode do mesmo frame, não de "cena parecida").

**Resultado da checagem** (relatório completo em `output/test_run_20260903_224733/leakage_report.txt`):

| Vídeo | Data no nome | Coincide com data de treino? | MSE mínimo encontrado | Imagem de treino mais próxima | Status |
| :--- | :---: | :---: | :---: | :--- | :---: |
| `front-dia-04-05.avi` | 04-05 | Não (nenhuma sessão em 05/04 ou 04/05) | 2546,8 | `Cam03_2025_04_27_08_16_51` frame 0136 | **Sem evidência de vazamento** |
| `front-noite-26-04.avi` | 26-04 | Sim (8 sessões em 26/04) | 1326,8 | `Cam03_2025_04_26_01_17_38` frame 0021 | **Sem evidência de vazamento** |
| `sup-dia-29-04.avi` | 29-04 | Sim (3 sessões em 29/04) | — (nenhum candidato passou no pré-filtro) | — | **Sem evidência de vazamento** |
| `sup-noite-29-04.avi` | 29-04 | Sim (3 sessões em 29/04) | 2355,4 | `Cam04_2025_04_29_02_05_12` frame 0067 | **Sem evidência de vazamento** |

**Conclusão: nenhum dos 4 vídeos de teste é uma duplicata confirmada de frames usados no treinamento.** Dois deles (`front-noite-26-04` e os dois `*-29-04`) compartilham a data de calendário com sessões de treino, mas o MSE mínimo encontrado (1326–2546) é da mesma ordem de grandeza do MSE entre dois frames diferentes *da própria sessão de treino* (750–1200) — ou seja, estatisticamente indistinguível de "não é o mesmo frame", muito acima do patamar de uma duplicata real (MSE ≈ 0–60). `front-dia-04-05.avi` nem sequer coincide em data com nenhuma sessão de treino.

### 7.2 Inferência YOLO nos Vídeos de Teste

Com a ausência de vazamento confirmada, a inferência foi executada com o `best.pt` atual (Fold 3, Seção 4.1), `conf=0.7`, `iou=0.7`, `imgsz=640`, na GPU RTX 5070. Resultado (`output/test_run_20260903_224733/relatorio_teste.txt`):

| Vídeo | Frames | Duração | FPS | Caixas Detectadas |
| :--- | :---: | :---: | :---: | :---: |
| `front-dia-04-05.avi` | 80 | 2,74s | 29,2 | 623 |
| `front-noite-26-04.avi` | 80 | 1,47s | 54,4 | 643 |
| `sup-dia-29-04.avi` | 80 | 1,59s | 50,4 | 532 |
| `sup-noite-29-04.avi` | 80 | 1,51s | 53,1 | 663 |

Tempo total da suíte (checagem de vazamento + inferência nos 4 vídeos): ≈ 7,3s de inferência pura (a checagem de vazamento em si leva mais tempo por ler milhares de imagens do HD).

### 7.3 Scripts e Correção Aplicada

* **`check_session_leakage.py`** — implementa a metodologia da Seção 7.1 isoladamente; pode ser rodado sozinho (`python check_session_leakage.py --report caminho.txt`) para reauditar novos vídeos de teste no futuro.
* **`run_video_tests.py`** — orquestrador único: roda a checagem de vazamento (Seção 7.1) e a inferência (Seção 7.2) em sequência, salvando tudo em `output/test_run_<timestamp>/` (ver comando na Seção 6.2, item 5). Substitui o uso direto de `test_videos.py` para validação de um novo `best.pt`.
* **Correção de path do Ultralytics:** `model.predict(project=..., name=...)` com caminho **relativo** é silenciosamente redirecionado pelo Ultralytics para o `runs_dir` global configurado em `~/.config/Ultralytics/settings.json` (`/home/vini/runs`), ignorando a pasta local `./output` — isso afetava também o `test_videos.py` legado. `run_video_tests.py` corrige isso resolvendo `project`/`name` para caminhos **absolutos** antes de chamar `model.predict(...)`, garantindo que os vídeos anotados e labels fiquem de fato dentro de `output/test_run_<timestamp>/`.

---

## 8. Comparativo no TCC: Baseline Overall vs. MoE (Mixture of Experts)

| Critério | Baseline Monolítico (YOLOv8 Overall) | Mixture of Experts (MoE Híbrido) |
| :--- | :--- | :--- |
| **Arquitetura** | Modelo único YOLOv8m treinado em todos os cenários combinados | Gating Network (EfficientNet-B0) + 4 Especialistas YOLOv8m dedicados |
| **Complexidade** | Simples de treinar e implantar (pipeline único) | Requer classificação de contexto prévia e chaveamento dinâmico |
| **Especialização** | Pesos médios compartilhados para características frontais, zenitais, diurnas e noturnas | Pesos hiper-especializados nas particularidades de iluminação e perspectiva |
| **Mitigação de Ruído** | Risco de interferência entre características diurnas (cor/textura) e noturnas (infravermelho/bordas) | Separação total de domínios visuais |
| **Desempenho** | Elevado em mAP50 global, com variações em condições limítrofes | Máxima aderência e sensibilidade contextual validada via XAI (Captum) |
