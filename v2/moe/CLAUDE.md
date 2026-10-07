# CLAUDE.md - Contexto Geral, Arquitetura e Organização do Projeto

Este arquivo concentra o contexto completo, histórico de desenvolvimento, arquitetura técnica e organização do projeto de **TCC: Detecção de Caprinos via Mixture of Experts (MoE)**.

---

## 1. Visão Geral do Projeto

* **Tema:** Detecção automatizada de caprinos em instalações de confinamento (capril) sob variações severas de iluminação e posicionamento de câmera.
* **Abordagem:** **Mixture of Experts (MoE)** híbrido combinando:
  1. **Gating Network (Router CNN):** Classifica o contexto/cenário da cena (iluminação e ângulo).
  2. **Especialistas (YOLOv8 Experts):** Modelos de detecção de objetos especializados para cada cenário específico.
* **Objetivo:** Superar as limitações de modelos monolíticos generalistas quando expostos a transições abruptas de luz (dia natural vs. infravermelho/noturno) e variações de perspectiva (frontal vs. superior/zenital).

---

## 2. Cenários e Especialistas (Mapeamento MoE)

O roteador classifica a entrada em 4 cenários distintos e ativa o especialista YOLOv8m correspondente:

| Classe CNN | Cenário / Descrição | Modelo YOLO Especialista | Pesos Consolidados |
| :--- | :--- | :--- | :--- |
| `class_1` | **Frente Dia** (Câmera Frontal, Iluminação Diurna) | YOLOv8m Specialist | `best-weights/best-front-dia.pt` |
| `class_2` | **Frente Noite** (Câmera Frontal, Infravermelho/Noite) | YOLOv8m Specialist | `best-weights/best-front-noite.pt` |
| `class_3` | **Cima Dia** (Câmera Superior, Iluminação Diurna) | YOLOv8m Specialist | `best-weights/best-sup-dia.pt` |
| `class_4` | **Cima Noite** (Câmera Superior, Infravermelho/Noite) | YOLOv8m Specialist | `best-weights/best-sup-noite.pt` |

---

## 3. Arquitetura e Componentes Técnicos

### 3.1 Gating Network (Classificador de Roteamento)
* **Arquitetura Base:** EfficientNet-B0 pré-treinada (`torchvision.models.efficientnet_b0`).
* **Camada de Saída:** Camada linear adaptada para 4 classes (`Linear(in_features, 4)`).
* **Entrada:** Imagens redimensionadas para $224 \times 224$ (com normalização ImageNet padrão).
* **Treinamento:** Otimizador Adam ($lr=0.001$), CrossEntropyLoss, Early Stopping com paciência de 7 épocas.
* **Pesos Salvos:** `goat-cnn/training_results/best_model.pth` e `modelo-final/cnn-weight/best_model.pth`.

### 3.2 Especialistas YOLOv8 (Detectores de Objetos)
* **Arquitetura Base:** YOLOv8 Médio (`yolov8m.pt`).
* **Validação Cruzada:** 5-Fold Cross-Validation com **Bloqueio Temporal Contínuo** (*Contiguous Time-Block Split / Temporal K-Fold*).
* **Prevenção de Vazamento de Dados (*Data Leakage*):** Imagens consecutivas do mesmo vídeo/sessão foram agrupadas cronologicamente; cada fold retém uma janela temporal contínua de 20% para validação e os outros 80% para treino. Uma zona de purga (*purge/embargo buffer*, default 15 frames de cada lado do bloco de validação) remove amostras próximas o suficiente da fronteira para serem quase-duplicatas visuais, e imagens cujo nome não bate com nenhum padrão de vídeo/frame reconhecido são excluídas do split (não é possível garantir sua ordem cronológica). Isso reduz o vazamento por adjacência temporal, mas não elimina vazamento de cena/sessão em especialistas com poucos vídeos-fonte independentes. Ver `kfold_audit_report.txt` em cada `yolo-8-*/` para os números reais por especialista.
  * **Atualização de dataset (2026-09-01/02):** o dataset bruto (`ds_front_dia`, `ds_front_noite`, `ds_sup_dia`, `ds_sup_noite`) cresceu **+65% em imagens** (5.204 → **8.604** imagens, 39.504 → **66.705** caixas), com frames extraídos diretamente de vídeo (sem passar pelo Roboflow). Isso exigiu duas correções em `generate_temporal_kfold.py`: busca de imagens passou de `glob()` (não recursivo) para `rglob()` (recursivo), pois os arquivos novos vinham em subpastas aninhadas (`images/images/`, `labels/labels/`); e uma nova regra de regex em `parse_filename_info()` para reconhecer o padrão `<prefixo>-frame_<N>.<ext>` desses frames sem sufixo Roboflow. As subpastas aninhadas foram achatadas em disco (0% de perda de arquivos). Ver `LEVANTAMENTO_DATASETS_20260901.md` para o levantamento completo do bug e da correção.
  * **Redução do risco de vazamento de cena em `front-noite`:** antes da atualização, `front-noite` tinha apenas 1 vídeo-fonte (pior caso de vazamento de cena/sessão entre treino e validação). Após a atualização passou a ter **13 vídeos-fonte independentes**, a maior diversidade entre os 4 especialistas. Vídeos-fonte por especialista pós-atualização: Frente Dia (8), Frente Noite (13), Cima Dia (6, sem alteração nesta rodada), Cima Noite (9).
* **Hiperparâmetros de Treinamento:** AutoBatch (`batch=-1`), resolução `imgsz=640`, otimizador padrão Ultralytics, early stopping automático com base em mAP50.

### 3.3 Explicabilidade e Interpretabilidade (XAI)
* **Método:** Estudo de Sensibilidade por Oclusão (*Occlusion Sensitivity*) via biblioteca `captum` (`goat-cnn/occlusion.py`).
* **Objetivo:** Interpretar a Gating Network gerando mapas de relevância e calor para validar se a CNN baseia as decisões de roteamento nas características do ambiente (ex: iluminação da cena, presença de barras e teto) em vez de artefatos espúrios.
* **Saídas:** Gráficos e painéis comparativos salvos em `goat-cnn/occlusion_results/`.

---

## 4. Estrutura e Organização do Diretório

```text
/home/vini/tcc/moe/
├── CLAUDE.md                         # [ESTE ARQUIVO] Contexto geral, arquitetura e documentação
├── LEVANTAMENTO_DATASETS_20260901.md # Levantamento do bug de detecção de imagens novas e preparação do retreino (2026-09-01)
├── .gitignore                        # Ignora pesos, checkpoints, caches e resultados grandes
├── generate_temporal_kfold.py        # Geração dos 5 folds temporais (purge buffer + limpeza de regex)
├── run_parallel_kfold_trainings.py   # Execução sequencial dos treinos dos 4 especialistas (cache RAM)
├── extract_best_metrics.py           # Extração das métricas dos melhores folds em cada especialista
├── tabela_resultados_yolo.md         # Tabela comparativa de métricas em Markdown (TCC)
├── tabela_resultados_yolo.csv        # Tabela de métricas em formato CSV
├── tabela_resultados_yolo.tex        # Tabela pronta formatada para LaTeX / Overleaf
├── weights/
│   └── yolov8m.pt                    # Pesos pré-treinados YOLOv8m (base compartilhada pelos 4 especialistas)
│
├── goat-cnn/                         # Módulo da Gating Network (EfficientNet-B0) — repositório git próprio
│                                      # (origin: github.com/ViniciusLatini/goat-cnn), versionado à parte
│   ├── create_ds.py                  # Script para criação/organização do dataset da CNN
│   ├── train.py                      # Script de treino da EfficientNet-B0 com Early Stopping
│   ├── test.py                       # Testes de inferência rápida da CNN
│   ├── occlusion.py                  # Análise de explicabilidade (XAI) via Occlusion Sensitivity
│   ├── info_ds.txt                   # Resumo estatístico da distribuição de classes do dataset
│   ├── dataset/                      # Imagens de treino/validação divididas por classes (class_1..4)
│   ├── test_imgs/                    # Imagens de referência para testes unitários
│   ├── training_results/             # Pesos `best_model.pth`, matriz de confusão e curvas de loss/acc
│   └── occlusion_results/            # Heatmaps gerados pelo estudo de oclusão
│
├── yolo-8-front-dia/                 # Treinamentos do especialista: Frente Dia
│   ├── data_fold_0..4.yaml           # Arquivos de configuração de dados para os 5 folds
│   ├── train_fold_0..4.txt           # Listagem de imagens para o conjunto de treino
│   ├── val_fold_0..4.txt             # Listagem de imagens para o conjunto de validação
│   ├── kfold_audit_report.txt        # Auditoria do split: exclusões por regex, purge, contagem por fold
│   ├── fold-metrics.txt              # Registro resumido dos melhores mAP50 obtidos
│   ├── README.dataset.txt / README.roboflow.txt  # Boilerplate do export Roboflow (só nesta pasta)
│   ├── runs/detect/                  # Checkpoints, pesos (`best.pt`, `last.pt`) e logs do YOLOv8 (treino vigente)
│   ├── runs/detect_archive_20260901/ # Backup do treino anterior (dataset pré-expansão), ~530MB, preservado para comparação no TCC
│   ├── kfold_audit_report_archive_20260901.txt  # Auditoria do split ANTES da correção/expansão do dataset
│   └── fold-metrics_archive_20260901.txt        # Métricas por fold ANTES da correção/expansão do dataset
│
├── yolo-8-front-noite/               # Treinamentos do especialista: Frente Noite (mesma estrutura + arquivos _archive_20260901)
├── yolo-8-sup-dia/                   # Treinamentos do especialista: Cima Dia (mesma estrutura + arquivos _archive_20260901)
├── yolo-8-sup-noite/                 # Treinamentos do especialista: Cima Noite (mesma estrutura + arquivos _archive_20260901)
│
└── modelo-final/                     # Pipeline Final Unificado de Predição MoE
    ├── README.md                     # Manual de uso detalhado do pipeline final
    ├── predicao_moe.py               # Script principal de predição (CNN Router + YOLO Especialista)
    ├── cnn-weight/
    │   └── best_model.pth            # Pesos treinados do classificador de cena EfficientNet-B0
    ├── best-weights/                 # Pesos consolidados dos melhores folds de cada especialista (retreinados 2026-09-02)
    │   ├── best-front-dia.pt
    │   ├── best-front-noite.pt
    │   ├── best-sup-dia.pt
    │   └── best-sup-noite.pt
    ├── best-weights copy/            # Backup dos pesos consolidados ANTERIORES (2026-08-17, dataset pré-expansão)
    ├── testes/                       # Vídeos de teste end-to-end do pipeline (1 por cenário: front-dia, front-noite, sup-dia, sup-noite)
    ├── resultados-testes-2026-09-03/ # Saída do teste end-to-end mais recente (ver seção 8), 1 subpasta por vídeo com anotado + `resumo.txt`
    └── resultados-front-dia/, resultados-front-noite/, resultados-sup-dia/, resultados-sup-noite/  # Saídas de testes individuais anteriores (2026-08-17, pesos antigos)
```

---

## 5. Resultados Consolidados dos Especialistas YOLOv8

**Estado do treinamento:** os 4 especialistas foram **retreinados em 2026-09-01/02** (20 folds, 5 por especialista, 300 épocas com early stopping automático, GPU RTX 5070) sobre o dataset expandido (seção 3.2, +65% de imagens). O treino anterior (dataset menor, concluído até 2026-08-17) foi preservado em `yolo-8-*/runs/detect_archive_20260901/` e nos arquivos `*_archive_20260901.txt` para comparação no texto do TCC. `python extract_best_metrics.py` foi reexecutado em 2026-09-03 sobre os novos `results.csv` de cada fold para confirmar a tabela abaixo, e os `best.pt` vencedores foram recopiados para `modelo-final/best-weights/` (pesos antigos preservados em `modelo-final/best-weights copy/`).

Tamanho do dataset por especialista após a expansão (imagens válidas usadas no split, ver `kfold_audit_report.txt`):

| Especialista | Imagens | Caixas (bboxes) | Vídeos-fonte | Fold 0 (treino/val) |
| :--- | ---: | ---: | :---: | :---: |
| Frente Dia | 2.022 | 16.185 | 8 | 1496/406 |
| Frente Noite | 2.001 | 16.008 | 13 | 1405/401 |
| Cima Dia | 2.549 | 18.258 | 6 | 1949/510 |
| Cima Noite | 2.032 | 16.254 | 9 | 1490/407 |
| **Total** | **8.604** | **66.705** | 36 | — |

Métricas obtidas pelos melhores folds selecionados após o K-Fold temporal (**dataset expandido, retreino 2026-09-02**):

| Modelo Especialista | Melhor Fold | Época | Precisão (P) | Revocação (R) | F1-Score | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8 Frontal Dia** | `fold_2` | 108 | 0.9966 | 0.9975 | 0.9971 | 0.9950 | 0.9015 |
| **YOLOv8 Frontal Noite** | `fold_3` | 81 | 0.9972 | 0.9968 | 0.9970 | 0.9950 | 0.8821 |
| **YOLOv8 Superior Dia** | `fold_4` | 50 | 0.9999 | 0.9995 | 0.9997 | 0.9950 | 0.8645 |
| **YOLOv8 Superior Noite** | `fold_5` | 13 | 0.9978 | 0.9972 | 0.9975 | 0.9950 | 0.8338 |

**Observação:** os valores acima coincidem numericamente com os do treino anterior (dataset menor). Isso não é um erro de reprocessamento — todos os folds de todos os especialistas, no dataset expandido, também saturam em mAP50 ≈ 0.9950 (variação entre folds na 3ª/4ª casa decimal; ver `fold-metrics.txt` por especialista), então o fold "vencedor" por especialista calhou de ser o mesmo índice de antes. Ou seja, os especialistas já operavam próximos do teto de mAP50 mesmo com bem menos dados; o ganho da expansão do dataset é sobretudo em **robustez/generalização** (mais vídeos-fonte independentes, especialmente em `front-noite`: 1 → 13), não em mAP50 no fold de validação.

---

## 6. Ambiente e Execução

### 6.1 Ambiente Conda
* **Nome do Ambiente:** `tcc`
* **Python Path:** `/home/vini/miniconda3/envs/tcc/bin/python`
* **Ativação:**
  ```bash
  conda activate tcc
  ```
* **Bibliotecas Chave:** `torch`, `torchvision`, `ultralytics` (YOLOv8), `captum`, `opencv-python`, `scikit-learn`, `matplotlib`, `seaborn`, `pillow`.

---

### 6.2 Comandos Principais

#### 1. Executar Predição MoE em Imagem ou Vídeo (Pipeline Completo)
```bash
cd /home/vini/tcc/moe/modelo-final
python predicao_moe.py --input <caminho_da_imagem_ou_video> --output resultados
```
* **Comportamento:**
  1. A CNN classifica o cenário da mídia através do primeiro frame.
  2. O modelo YOLO especialista correspondente é carregado dinamicamente.
  3. A inferência é processada e o vídeo/imagem anotado é salvo em `resultados/predicao_moe/`.
  4. Um relatório de contagem e métricas é exportado em `resultados/predicao_moe/resumo.txt`.

#### 2. Gerar Divisões do K-Fold Temporal
```bash
cd /home/vini/tcc/moe
python generate_temporal_kfold.py --purge-frames 15
```
*Gera `kfold_audit_report.txt` em cada `yolo-8-*/` com exclusões por regex, purge aplicado e contagem por fold.*

#### 3. Treinar Especialistas YOLOv8 em K-Fold
```bash
cd /home/vini/tcc/moe
python run_parallel_kfold_trainings.py --epochs 300
```

#### 4. Extrair Métricas e Atualizar Tabelas do TCC
```bash
cd /home/vini/tcc/moe
python extract_best_metrics.py
```
*Atualiza automaticamente `tabela_resultados_yolo.csv`, `tabela_resultados_yolo.md` e `tabela_resultados_yolo.tex`.*

#### 5. Executar Análise de Explicabilidade (XAI)
```bash
cd /home/vini/tcc/moe/goat-cnn
python occlusion.py
```

---

## 7. Teste de Ponta a Ponta do Pipeline MoE (2026-09-03)

Após o retreino dos especialistas (seção 5), o pipeline completo (`modelo-final/predicao_moe.py`) foi validado de ponta a ponta contra os 4 vídeos de referência em `modelo-final/testes/` (um por cenário), usando os pesos atuais (`best-weights/`, atualizados em 2026-09-02).

* **Comando usado (repetido para cada vídeo, saída isolada por vídeo):**
  ```bash
  cd /home/vini/tcc/moe/modelo-final
  python predicao_moe.py --input testes/<video>.avi --output resultados-testes-2026-09-03/<video>
  ```
* **Saída:** cada vídeo gera sua própria subpasta em `resultados-testes-2026-09-03/<video>/predicao_moe/`, contendo o vídeo anotado e o `resumo.txt` (classe/confiança da CNN, especialista YOLO ativado, tempo de inferência).

Resultado — em todos os 4 vídeos a CNN roteou para o especialista correto:

| Vídeo de Teste | Cenário Roteado (CNN) | Confiança CNN | Especialista YOLO Ativado |
| :--- | :--- | :---: | :--- |
| `front-dia-04-05.avi` | Frente Dia ✓ | 99.32% | `best-front-dia.pt` |
| `front-noite-26-04.avi` | Frente Noite ✓ | **62.67%** | `best-front-noite.pt` |
| `sup-dia-29-04.avi` | Cima Dia ✓ | 99.38% | `best-sup-dia.pt` |
| `sup-noite-29-04.avi` | Cima Noite ✓ | 99.99% | `best-sup-noite.pt` |

**Ponto de atenção:** a confiança da CNN roteadora no cenário `front-noite` (62.67%) ficou bem abaixo dos demais (>99%), embora tenha classificado a classe correta. Vale investigar o(s) frame(s) inicial(is) desse vídeo específico (possível ambiguidade visual entre `front-noite` e outro cenário no primeiro frame usado para classificação) antes de reportar esse caso no texto do TCC.

Um teste anterior (2026-08-17, pesos antigos pré-expansão do dataset) está preservado em `modelo-final/resultados-front-dia/`, `resultados-front-noite/`, `resultados-sup-dia/` e `resultados-sup-noite/`, cada um com seu próprio `resumo.txt`.

---

## 8. Histórico do Desenvolvimento e Conquistas

1. **Estruturação do Dataset e Roteador:**
   - Criação do dataset de cenários com 1360 imagens balanceadas e estratificadas em 4 classes.
   - Treinamento da EfficientNet-B0 com transfer learning, atingindo convergência e alta acurácia na separação de contexto (dia/noite/frontal/superior).
2. **Mitigação de Vazamento de Dados (Temporal Leakage):**
   - Implementação do algoritmo de *Contiguous Time-Block Split*, garantindo que frames consecutivos da mesma sessão não se misturassem entre treino e teste.
3. **Treinamento e Otimização dos Especialistas YOLOv8m:**
   - Treinamento automatizado dos 5 folds para os 4 cenários na GPU RTX 5070 com AutoBatch.
   - Seleção dos melhores pesos (`best.pt`) por fold com base no mAP50 e mAP50-95.
4. **Pipeline Integrado de Produção:**
   - Consolidação dos pesos em `modelo-final/best-weights` e desenvolvimento do `predicao_moe.py` para inferência de ponta a ponta com suporte a streaming de vídeo e imagens individuais.
5. **Validação e Explicabilidade:**
   - Análise de sensibilidade por oclusão com Captum para fundamentação teórica e discussão dos resultados no texto do TCC.
6. **Teste de ponta a ponta com pesos antigos (2026-08-17):**
   - Primeira validação do pipeline `predicao_moe.py` contra 1 vídeo por cenário, salvando resultado e `resumo.txt` em `modelo-final/resultados-<cenário>/`.
7. **Expansão e correção do dataset de treino dos especialistas (2026-09-01):**
   - Identificado e corrigido bug em `generate_temporal_kfold.py`: busca não recursiva (`glob`) não enxergava imagens novas depositadas em subpastas aninhadas (`images/images/`, `labels/labels/`); trocado para `rglob`.
   - Adicionada nova regra de regex em `parse_filename_info()` para reconhecer frames extraídos direto de vídeo (sem sufixo Roboflow), padrão `<prefixo>-frame_<N>.<ext>`.
   - Subpastas aninhadas achatadas em disco sem perda de arquivos; dataset total cresceu de 5.204 para **8.604 imagens** (+65%), concentrado em `front_dia`, `front_noite` e `sup_noite`.
   - `front-noite` deixou de ser o especialista de maior risco de vazamento de cena (1 → 13 vídeos-fonte independentes).
   - Resultados/pesos do treino anterior arquivados (`runs/detect_archive_20260901/`, `*_archive_20260901.txt`) antes de sobrescrever, para permitir comparação antes/depois no texto do TCC. Ver `LEVANTAMENTO_DATASETS_20260901.md` para o levantamento completo.
8. **Retreino completo dos especialistas YOLOv8 sobre o dataset expandido (2026-09-01/02):**
   - 20 folds (5 por especialista × 4 especialistas) retreinados via `run_parallel_kfold_trainings.py --epochs 300` na GPU RTX 5070.
   - `extract_best_metrics.py` reexecutado (2026-09-03) para confirmar os melhores folds sobre os novos `results.csv`; tabelas `tabela_resultados_yolo.{md,csv,tex}` regeneradas.
   - mAP50 dos melhores folds manteve-se saturado em ≈0.9950 em todos os especialistas (mesmo com +65% de dados) — ganho do retreino é de robustez/generalização, não de mAP50 de validação (ver seção 5).
   - `best.pt` vencedores recopiados para `modelo-final/best-weights/`; pesos antigos preservados em `modelo-final/best-weights copy/`.
9. **Reteste de ponta a ponta com pesos retreinados (2026-09-03):**
   - Pipeline `predicao_moe.py` reexecutado contra os 4 vídeos de `modelo-final/testes/`, cada um salvo isoladamente em `modelo-final/resultados-testes-2026-09-03/<vídeo>/` com vídeo anotado + `resumo.txt`.
   - Roteamento da CNN acertou o cenário correto nos 4 vídeos; confiança da CNN ficou anormalmente baixa (62.67%) no vídeo `front-noite`, sinalizado como ponto a investigar (ver seção 7).
