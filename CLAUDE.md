# CLAUDE.md — TCC: Detecção de Caprinos em Confinamento via Mistura de Especialistas (MoE)

Documento-mestre do repositório. Consolida o histórico completo do projeto, onde está cada coisa, os números oficiais e o estado da escrita da monografia. Os `CLAUDE.md` específicos (`v2/`, `v2/moe/`, `v2/yolo-8-overall/`) têm o detalhe técnico de cada módulo; este arquivo é o ponto de entrada.

- **Autor:** Vinícius Latini Gonçalves (UFJF — Ciência da Computação)
- **Orientação:** Luciana Conceição D. Campos (UFJF); coorientação/parceria: Érica da Costa Reis Carvalho (UFSJ)
- **Parceiros UFSJ (dados/rotulação):** Gabriel Vaz Bernardini Ferreira, Felipe Girardi Siqueira, Acir Moreno Soares Júnior (Zootecnia)
- **Projeto guarda-chuva (UFSJ):** "Inteligência artificial aplicada à detecção e classificação comportamental de cabras em confinamento (raça Saanen)". O TCC cobre **somente o módulo de detecção**.

---

## 1. Mapa do repositório

```text
/home/vini/tcc/
├── CLAUDE.md               # [ESTE ARQUIVO]
├── documentacao_tcc/       # >>> TEXTO DO TCC (monografia) — é aqui que se escreve <<<
│   └── Template_SBC/
│       ├── common.tex                  # macros de revisão: \entra, \sai, \rever, \todo, \etal
│       ├── sbc_template.{odt,pdf}      # template SBC original (referência)
│       └── template-latex/
│           ├── sbc-template.tex        # ARQUIVO PRINCIPAL DO TCC
│           ├── sbc-template.bib        # bibliografia do TCC
│           ├── sbc-template.sty, sbc.bst, caption2.sty
│           └── images/                 # figuras (hoje só MoE.jpeg; as demais faltam localmente)
├── v2/                     # >>> VERSÃO FINAL DO PROJETO (arquitetura MoE) <<<
│   ├── CLAUDE.md           # diagnóstico da revisão do artigo (5 apontamentos do revisor)
│   ├── artigo/             # artigo ENMC/ECTM 2026 revisado (main_revisado.tex) — fonte mais atualizada do texto
│   ├── posterEvento/       # pôster do evento (LaTeX)
│   ├── moe/                # gating EfficientNet-B0 + 4 especialistas YOLOv8m + pipeline final
│   └── yolo-8-overall/     # baseline monolítico (1 YOLOv8m em todos os cenários)
├── sheep/                  # PoC: YOLOv8m COCO zero-shot, classe 18 (sheep)
├── yolo-8-1000/            # experimento legado: 1º dataset próprio (~1.590 imgs), K-Fold aleatório
├── yolo-8-v2/              # experimento legado: dataset próprio ampliado (~1.840 imgs), K-Fold aleatório
└── teste.py                # checagem de CUDA
```

Ambiente: conda `tcc` (`/home/vini/miniconda3/envs/tcc/bin/python`), GPU NVIDIA RTX 5070 12 GB. Libs: torch, torchvision, ultralytics, captum, opencv-python, scikit-learn, pandas, matplotlib, seaborn.

---

## 2. Linha do tempo do desenvolvimento

1. **Prova de conceito (`sheep/`)** — YOLOv8m pré-treinado no COCO, sem ajuste, filtrando a classe `sheep` (COCO não tem `goat`). Detectou só 2 cabras, confianças 0,27 e 0,48; testes em dia/noite e outras vistas também falharam → motivou o dataset próprio. Figura: `v2/artigo/figures/sheep-test.png`.
2. **Primeiro dataset próprio + YOLOv8 (`yolo-8-1000/`)** — frames extraídos aleatoriamente, rotulados no Makesense.ai por voluntários UFJF/UFSJ (guia de rotulação em 5 etapas, classe única `goat`). Treino 80/20 e K-Fold (5) **aleatório por imagem**. Números altíssimos (mAP50 ≈ 0,995; mAP50-95 até 0,955) — hoje sabidamente **inflados por vazamento temporal** (frames vizinhos em treino e validação).
3. **Dataset segmentado por cenário (UFSJ) e dataset via API (`yolo-8-v2/`, EasyLabel)** — alunos da UFSJ separaram as imagens por perspectiva × iluminação e criaram a ferramenta/API EasyLabel (https://github.com/Gabriel0Bernardini/EasyLabel) para rotulação padronizada. Treinos especializados sugeriram ganho da especialização → origem da ideia MoE.
4. **Arquitetura MoE (`v2/moe/`)** — gating EfficientNet-B0 classifica o cenário; 4 especialistas YOLOv8m (frontal-dia, frontal-noite, superior-dia, superior-noite). Roteamento top-1 (rígido), feito **uma vez por vídeo, no 1º frame**.
5. **Revisão do artigo (v2/CLAUDE.md)** — revisor apontou: (1) escopo "classificação" inconsistente; (2) dataset não descrito; (3) alegação de Raspberry Pi sem medição; (4) vazamento no K-Fold; (5) falta de baseline monolítico treinado. Respostas: reframe para detecção; caracterização do dataset; remoção das alegações de hardware; **K-Fold temporal com purge**; **baseline `yolo-8-overall`**.
6. **Correção + expansão do dataset (2026-09-01)** — bug de `glob` não recursivo e novo padrão de nome (`-frame_N`) corrigidos em `generate_temporal_kfold.py`; dataset 5.204 → **8.604 imagens** (+65%). Ver `v2/moe/LEVANTAMENTO_DATASETS_20260901.md`.
7. **Retreino dos 4 especialistas (2026-09-01/02)** e **treino do baseline (2026-09-03, ~15 h)** sob o mesmo protocolo.
8. **Teste ponta a ponta em 4 vídeos independentes (2026-09-03)** — checagem de vazamento de sessão (aHash + MSE) + contagem de caixas MoE vs. baseline.

---

## 3. Números oficiais (usar SEMPRE estes no texto)

### 3.1 Aquisição
- 4 câmeras fixas (Cam01–Cam04); perspectivas **frontal** e **superior (zenital)**; iluminação **diurna** (luz natural) e **noturna** (infravermelho).
- Resolução nativa 1280×960, **8 FPS**, **36 sessões** de gravação, **7 dias** de aquisição, rebanho de **8 cabras Saanen** na mesma baia.
- Entrada dos detectores: 640×640. Entrada da gating: 224×224 (normalização ImageNet).

### 3.2 Dataset de detecção (classe única `goat`)

| Cenário | Câmeras | Sessões | Imagens | Caixas | Caixas/img |
|---|---|---:|---:|---:|---:|
| Frontal Dia | Cam01, Cam03 | 8 | 2.022 | 16.185 | 8,01 |
| Frontal Noite | Cam03 | 13 | 2.001 | 16.008 | 8,00 |
| Superior Dia | Cam02, Cam04 | 6 | 2.549 | 18.258 | 7,16 |
| Superior Noite | Cam02, Cam04 | 9 | 2.032 | 16.254 | 8,00 |
| **Total** | Cam01–Cam04 | **36** | **8.604** | **66.705** | 7,75 |

### 3.3 Dataset da gating network
1.360 imagens, divisão estratificada 80/20 → 1.088 treino / 272 validação.
Treino por classe: class_1 (Frontal Dia) 448 · class_2 (Frontal Noite) 64 · class_3 (Superior Dia) 384 · class_4 (Superior Noite) 192 (validação na mesma proporção: 112/16/96/48).
**Atenção:** é estratificado, mas **não balanceado** — o artigo diz "balanceado"; corrigir no TCC.

### 3.4 Protocolo de validação (Temporal K-Fold)
K = 5; imagens agrupadas por sessão (a partir do nome do arquivo), ordenadas no tempo; cada fold reserva um bloco **contíguo de 20%** da linha do tempo de cada sessão para validação; **purge buffer de 15 frames** em cada borda; 0% de sobreposição exata treino/val (auditado). Conjunto unificado: 4.234 amostras removidas pelo purge (soma dos folds); tamanhos treino/val — F0 6.340/1.724, F1 5.847/1.720, F2 5.803/1.721, F3 5.847/1.720, F4 6.345/1.719.
**Limitação declarada:** blocos de uma mesma sessão podem estar em treino e validação (não é Group K-Fold por sessão/dia).

### 3.5 Hiperparâmetros
YOLOv8m (pré-treino COCO), 300 épocas máx., `patience=30`, `imgsz=640`, `batch=-1` (AutoBatch), `cache='ram'`. Gating: EfficientNet-B0 (torchvision, ImageNet), cabeça `Linear(…, 4)`, Adam lr=0,001, CrossEntropy, até 50 épocas, early stopping paciência 7 (parou na época 9).

### 3.6 Resultados
**Especialistas (melhor fold de cada):**

| Especialista | Fold | Época | P | R | F1 | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Frontal Dia | 2 | 108 | 0,9966 | 0,9975 | 0,9971 | 0,9950 | 0,9015 |
| Frontal Noite | 3 | 81 | 0,9972 | 0,9968 | 0,9970 | 0,9950 | 0,8821 |
| Superior Dia | 4 | 50 | 0,9999 | 0,9995 | 0,9997 | 0,9950 | 0,8645 |
| Superior Noite | 5 | 13 | 0,9978 | 0,9972 | 0,9975 | 0,9950 | 0,8338 |

**Baseline monolítico (média dos 5 folds):** P 0,9948 · R 0,9957 · F1 0,9953 · mAP50 0,9948 · mAP50-95 0,8773 (melhor fold: 3, mAP50-95 0,8965).

**Gating:** P = R = F1 = 1,00 nas 272 imagens de validação. Oclusão (Captum; Zeiler & Fergus) mostra foco em paredes e cercado da baia (contexto), não nos animais.

**Vídeos de teste independentes** (4 vídeos × 80 frames; sem vazamento confirmado por aHash+MSE; `conf=0,7`, `iou=0,7`; esperado 8×80 = 640 caixas):

| Cenário | Baseline | MoE | Esperado |
|---|---:|---:|---:|
| Frontal Dia | 623 | 640 | 640 |
| Frontal Noite | 643 | 640 | 640 |
| Superior Dia | 532 | 640 | 640 |
| Superior Noite | 663 | 640 | 640 |

Baseline: subdetecção em Superior Dia (5–7 caixas/frame, nunca 8); sobredetecção em Superior Noite (26/80 frames com >8, até 10) e Frontal Noite (4/80), caixas duplicadas com IoU até 0,64 < limiar NMS 0,7. MoE: confiança das 2.560 caixas entre 0,74 e 0,96 (média 0,91). Roteamento correto nos 4 vídeos (confiança da CNN: 99,32% / **62,67%** frontal-noite / 99,38% / 99,99%).

**Mensagem honesta que o TCC deve sustentar:** o MoE **não** supera o baseline em mAP agregado (ambos saturam mAP50 ≈ 0,995; mAP50-95 próximos). O ganho está em **estabilidade da contagem em vídeo**, **modularidade/escalabilidade**, **isolamento de falhas** e **interpretabilidade do roteamento**.

---

## 4. Texto do TCC — estado e diretrizes

**Arquivo:** `documentacao_tcc/Template_SBC/template-latex/sbc-template.tex` (classe `article` + `sbc-template.sty`; "capítulos" = `\section`). Compilado a partir de `documentacao_tcc/` (há caminhos como `Template_SBC/common` e `Template_SBC/template-latex/images/...`), provavelmente no Overleaf.

**Estrutura planejada (anunciada na seção "Organização do Trabalho" do cap. 1):**
1. Introdução — **ESCRITO (2026-09-24)**: contextualização, motivação/problema, proposta, objetivos (geral + específicos), justificativa, escopo e delimitação, contribuições, organização. As antigas seções "Objetivos" e "Justificativa" foram incorporadas ao capítulo.
2. Fundamentação Teórica — a fazer (CNNs, detecção de objetos/YOLO, métricas P/R/F1/IoU/mAP, EfficientNet, MoE, validação cruzada e vazamento, oclusão/XAI).
3. Trabalhos Relacionados — a fazer (visão computacional em pecuária de precisão; detecção/contagem de caprinos; MoE em visão).
4. Materiais e Métodos — a fazer (hoje ainda é o texto antigo "Metodologia", com K-Fold aleatório e tabelas legadas → reescrever com o protocolo temporal e o dataset da seção 3).
5. Resultados e Discussão — a fazer (PoC; gating + oclusão; especialistas; baseline; contagem em vídeo).
6. Conclusão — a fazer.

**Diretrizes de escrita:**
- Português do Brasil, registro acadêmico; termos em inglês em `\textit{}`; classe em `\texttt{goat}`.
- Terminologia fixa: "Mistura de Especialistas (MoE)", "rede de roteamento (*gating network*)", "especialistas", "baseline monolítico", cenários **Frontal Dia / Frontal Noite / Superior Dia / Superior Noite** (no código: front/sup, class_1..4; no artigo às vezes "Cima").
- Escopo = **detecção de classe única**. Não prometer classificação comportamental, identificação individual, rastreamento, nem desempenho em Raspberry Pi/hardware embarcado (sem medições).
- A gating classifica o **cenário de captura**, nunca o animal.
- Não reutilizar as tabelas/números legados do texto antigo (K-Fold aleatório: mAP50-95 0,9316; "dataset por API": 0,9063; 114 épocas etc.) como resultado final — no máximo como histórico que motivou a correção do protocolo.
- Marcar pendências com `\todo{...}` (definido em `common.tex`).
- Toda citação nova vai em `sbc-template.bib`; conferir metadados reais antes de adicionar.

**Pendências conhecidas no documento (fora do cap. 1):**
- Título e resumo ainda falam em "detecção e classificação" → alinhar ao artigo ("…detecção de caprinos em ambiente confinado").
- Preâmbulo carrega `\usepackage[utf8]{inputenc}` **e** `\usepackage[latin1]{inputenc}`; o arquivo é UTF-8 → remover a linha `latin1`.
- Figuras referenciadas (`sheep-test.png`, `tutorial_p*.jpg`, `loss-200.png`, `map50_k-fold.png`…) não estão em `images/` localmente; as figuras finais estão em `v2/artigo/figures/`.
- Investigar a confiança baixa (62,67%) da gating no vídeo frontal-noite antes de discutir.
