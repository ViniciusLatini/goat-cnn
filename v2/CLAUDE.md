# CLAUDE.md - Contexto Geral, Situação Atual e Diretrizes da Revisão

Este arquivo consolida a situação atual do projeto de TCC (**Detecção de Caprinos via Mixture of Experts - MoE**), mapeando a estrutura do repositório, o status dos experimentos e o plano de ação detalhado para responder aos apontamentos da revisão do artigo.

---

## 1. Mapeamento dos Diretórios e Estrutura do Projeto

A raiz do repositório (`/home/vini/tcc/v2/`) está dividida em 3 pilares principais:

```text
/home/vini/tcc/v2/
├── artigo/           # Manuscrito do paper na versão enviada anteriormente (LaTeX, figuras, referências)
├── moe/              # Arquitetura proposta (Gating Network EfficientNet-B0 + 4 Especialistas YOLOv8m)
└── yolo-8-overall/   # Baseline Monolítico Generalista (YOLOv8m único treinado no dataset unificado)
```

### Detalhamento dos Módulos:
1. **`artigo/`**: Contém o texto-fonte em LaTeX (`main.tex`, `main_revisado.tex`, `common.tex`, figuras e bib). É o documento que precisa ser modificado para incorporar a nova metodologia, os novos resultados empíricos e corrigir as inconsistências apontadas pelos revisores.
2. **`moe/`**: Contém o modelo Mixture of Experts. Consiste em:
   - **Gating Network (CNN Router):** `goat-cnn/` (EfficientNet-B0) que classifica a cena em 4 cenários (Frente Dia, Frente Noite, Cima Dia, Cima Noite).
   - **Especialistas YOLOv8m:** Pastas `yolo-8-front-dia`, `yolo-8-front-noite`, `yolo-8-sup-dia`, `yolo-8-sup-noite`.
   - **Pipeline Unificado:** `modelo-final/predicao_moe.py`.
3. **`yolo-8-overall/`**: Contém o modelo monolítico generalista (único YOLOv8m treinado na união de todos os 4 cenários). Serve como o **baseline oficial** de comparação direta contra o MoE.

---

## 2. Diagnóstico das Ressalvas do Revisor e Status de Resolução

Abaixo está o mapeamento dos 5 comentários do revisor, acompanhado do diagnóstico atual no repositório e das ações necessárias:

| # | Apontamento do Revisor | Diagnóstico & Status Atual no Repositório | Ação Necessária no Artigo / Código |
|---|---|---|---|
| **1** | **Objetivo & Escopo Inconsistentes:** Promete classificação de animais e "análise de diferentes modelos", mas executa detecção de classe única (`goat`) com um só detector. A classificação opera sobre a cena de captura. | **Pendente de ajuste no artigo.** A arquitetura realiza detecção da classe `goat` auxiliada por classificação de contexto/cenário de captura pela Gating Network. | Reframe do título, resumo e objetivo em `artigo/main_revisado.tex`. Alinhar a promessa ao que foi efetivamente construído: detecção robusta de caprinos em confinamento usando arquitetura modular (MoE). |
| **2** | **Ausência de Descrição do Dataset:** Falta número total de imagens, instâncias anotadas, número de animais, dias, resolução, FPS, amostragem e splits. | **Resolvido em dados.** Dataset expandido e auditado (8.604 imagens, 66.705 caixas anotadas, 36 sessões/vídeos, câmeras Cam01–Cam04). Documentado em `moe/LEVANTAMENTO_DATASETS_20260901.md` e `yolo-8-overall/CLAUDE.md`. | Adicionar subseção e Tabela detalhada em `artigo/` com a caracterização completa do dataset (imagens, bboxes, resoluções, ângulos, iluminação, sessões). |
| **3** | **Alegações de Hardware Embarcado (Raspberry Pi):** Citado no abstract sem nenhuma medição no hardware-alvo. | **Pendente de medição / ajuste de redação.** Não há benchmarks de hardware embarcado no texto atual. | Realizar medições de latência/FPS/VRAM/RAM dos modelos (MoE vs Overall) ou remover alegações não comprovadas de viabilidade em Raspberry Pi do abstract/intro. |
| **4** | **Vazamento de Dados (Data Leakage em K-Fold):** Frames contínuos de vídeos idênticos colocados em treino e validação via K-Fold padrão (Tabela 1 mede memorização). | **RESOLVIDO NO CÓDIGO.** Desenvolvido e aplicado o *Contiguous Time-Block Split / Temporal GroupKFold* (`generate_temporal_kfold.py`) com Purge Buffer de 15 frames entre treino/val para garantir separação de sessões temporais. | Atualizar a seção de Metodologia em `artigo/` explicando o GroupKFold por bloco temporal/vídeo, o purging de borda e a eliminaçao do vazamento. |
| **5** | **Baseline Monolítico Ausente:** Comparação anterior usou apenas YOLOv8 zero-shot com classe `sheep` do COCO (sem valor metodológico). Faltava YOLOv8m treinado no dataset unificado. | **RESOLVIDO NO CÓDIGO.** Treinamento do `yolo-8-overall` em 5-Fold temporal completo concluído em **2026-09-03** sobre as 8.604 imagens (mAP50 médio: 0.9948, mAP50-95 médio: 0.8773). | Substituir a comparação antiga pela tabela oficial comparando `yolo-8-overall` (baseline unificado) vs `moe` (especialistas) sob o mesmo protocolo de validação. |

---

## 3. Detalhamento dos Resultados do Baseline Monolítico (`yolo-8-overall`)

Concluído em **2026-09-03** (GPU RTX 5070, ~15h de treinamento total sequencial nos 5 Folds temporais sobre `ds-overall-raw/` de 8.604 imagens):

| Fold | Época de Pico | Precisão (P) | Revocação (R) | F1-Score | mAP50 | mAP50-95 | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 96 | 0.9914 | 0.9938 | 0.9926 | 0.9944 | 0.8700 | Estável |
| **Fold 2** | 60 | 0.9945 | 0.9942 | 0.9943 | 0.9948 | 0.8707 | Estável |
| **Fold 3** | 54 | 0.9977 | 0.9986 | 0.9982 | 0.9950 | 0.8965 | **Melhor Geral** (`best.pt`) |
| **Fold 4** | 33 | 0.9967 | 0.9971 | 0.9969 | 0.9950 | 0.8839 | Estável |
| **Fold 5** | 25 | 0.9939 | 0.9948 | 0.9943 | 0.9949 | 0.8653 | Estável |
| **Média** | — | **0.9948** | **0.9957** | **0.9953** | **0.9948** | **0.8773** | **Alta Convergência** |

---

## 4. Plano de Ação para a Revisão do Artigo (`artigo/`)

Para submeter a versão final corrigida do artigo, as seguintes etapas devem ser seguidas:

1. **Ajuste de Título, Abstract e Introdução (`main_revisado.tex`):**
   - Corrigir a definição de objetivo: explicitar a detecção de caprinos (classe única `goat`) em capril utilizando uma abordagem Mixture of Experts (MoE) adaptativa às condições de captura (iluminação dia/noite e perspectiva frente/superior).
   - Remover menções sem respaldo experimental a Raspberry Pi / hardware embarcado no Abstract e Introdução (ou adicionar a seção de benchmark de hardware).

2. **Seção de Caracterização do Dataset:**
   - Adicionar tabela com dados quantitativos do dataset: 8.604 imagens, 66.705 anotações, 36 sessões de gravação em vídeo, resoluções, taxas de amostragem, distribuição por câmera (`Cam01` a `Cam04`) e por cenário (Frente Dia, Frente Noite, Cima Dia, Cima Noite).

3. **Seção de Metodologia e Prevenção de Vazamento (GroupKFold Temporal):**
   - Descrever detalhadamente a estratégia de validação cruzada baseada em blocos temporais continuos / agrupamento por vídeo/sessão (*Contiguous Time-Block Split*).
   - Explicar a margem de segurança (*purge buffer*) de 15 frames para eliminar a contaminação por dependência temporal entre vizinhos próximos.

4. **Seção de Resultados e Discussão:**
   - Remover completamente a comparação antiga com YOLOv8-COCO zero-shot (`sheep`).
   - Apresentar a comparação rigorosa entre o Baseline Monolítico (`yolo-8-overall`, mAP50=0.9948, mAP50-95=0.8773) e os Especialistas MoE (`moe`).
   - Discutir os trade-offs: eficácia em cenários específicos, tempo de inferência, custo computacional e explicabilidade (XAI via oclusão na Gating Network).

---

## 5. Instruções de Uso para Assistentes de IA

Ao interagir neste repositório:
- Respeite a divisão de escopo entre `artigo/`, `moe/` e `yolo-8-overall/`.
- Ao sugerir edições no arquivo LaTeX (`artigo/main_revisado.tex`), consulte sempre os dados numéricos reais consolidados em `moe/CLAUDE.md` e `yolo-8-overall/CLAUDE.md`.
- Garanta que qualquer alteração metodológica no texto reflita com precisão os scripts Python executados (`generate_temporal_kfold.py`, `run_kfold_trainings.py`, etc.).
