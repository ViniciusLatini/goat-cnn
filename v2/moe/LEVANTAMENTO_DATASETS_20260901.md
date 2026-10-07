# Levantamento do Estado Atual dos Datasets e Preparação para Novo Treinamento Completo

**Data do levantamento:** 2026-09-01
**Origem dos dados:** `/run/media/vini/061a1dd7-8e19-49bb-a45c-87c4603733ea/vini/ds/` (pastas `ds_front_dia`, `ds_front_noite`, `ds_sup_dia`, `ds_sup_noite`, cada uma com subpastas `images/` e `labels/`), confirmado pelo usuário como o caminho correto e atualizado.

---

## 1. Bug encontrado e corrigido: imagens novas não estavam sendo detectadas

Na primeira passada deste levantamento, o script `generate_temporal_kfold.py` reportou os mesmos totais de 13/08/2026 (1.172 / 451 / 2.549 / 1.032 imagens), mesmo após a atualização do dataset. O usuário apontou que `ds_front_noite`, por exemplo, deveria ter **2.002 amostras**. Investigação encontrou dois problemas:

### 1.1 Estrutura de pastas aninhada
As imagens/labels novas tinham sido depositadas em uma subpasta aninhada:
```
ds_front_dia/images/images/*.jpg      (850 arquivos novos)
ds_front_dia/labels/labels/*.txt      (850 arquivos novos)
ds_front_noite/images/images/*.jpg    (1.550 arquivos novos)
ds_front_noite/labels/labels/*.txt    (1.550 arquivos novos)
ds_sup_noite/images/images/*.jpg      (1.000 arquivos novos)
ds_sup_noite/labels/labels/*.txt      (1.000 arquivos novos)
```
`ds_sup_dia` não tinha essa subpasta (não recebeu atualização desta vez). O `generate_temporal_kfold.py` fazia uma busca **não recursiva** (`images_dir.glob('*.jpg')`), então todo esse conteúdo aninhado era completamente invisível ao script.

### 1.2 Novo padrão de nome de arquivo, sem export do Roboflow
Os arquivos novos vêm direto de extração de frames de vídeo, sem passar pelo pipeline do Roboflow: sem sufixo `.rf.<hash>` e com extensão real `.jpg` (ex.: `Cam03_2025_04_25_16_17_49_capril-frame_0000.jpg`), em vez do padrão mangled do Roboflow (`..._frame_0000_jpg.rf.<hash>.jpg`). Nenhuma das 4 regras de regex existentes em `parse_filename_info()` reconhecia esse formato — mesmo com busca recursiva, essas imagens cairiam no balde de "nome fora do padrão" e seriam excluídas do split.

### 1.3 Um terceiro problema, mais sério, que a estrutura aninhada esconderia até o treino falhar
Testei como o Ultralytics resolve o caminho do label a partir do caminho da imagem (`img2label_paths`) para o caso aninhado: ele substitui apenas a **última** ocorrência de `/images/` no caminho, então `.../images/images/x.jpg` viraria `.../images/labels/x.jpg` — um caminho que **não existe** (os labels novos estavam fisicamente em `.../labels/labels/x.jpg`). Se essa estrutura tivesse sido mantida e o bug de detecção "corrigido" apenas com busca recursiva, o treino teria rodado silenciosamente sem nenhum label para 850–1.550 imagens novas por especialista (tratando-as como background/sem objeto), corrompendo o treino sem erro explícito.

### Correção aplicada
1. **`generate_temporal_kfold.py`**: busca de imagens trocada de `glob()` para `rglob()` (recursiva) — proteção permanente contra essa classe de erro no futuro.
2. **`generate_temporal_kfold.py`**: nova regra de regex em `parse_filename_info()` para o padrão `<prefixo>-frame_<N>.<jpg|jpeg|png>` (frames extraídos diretamente, sem Roboflow).
3. **Dataset no disco**: as subpastas aninhadas foram **achatadas** (conteúdo de `images/images/` movido para `images/`, `labels/labels/` movido para `labels/`, pastas vazias removidas), eliminando também o risco do item 1.3 e deixando a estrutura consistente com o resto do dataset (`images/arquivo.jpg` + `labels/arquivo.txt`, sem aninhamento). Validado com `ultralytics.data.utils.img2label_paths` que o mapeamento imagem→label agora resolve corretamente.

Nenhum arquivo foi perdido: 0% de sobreposição de nomes entre o conteúdo aninhado e o conteúdo já existente em cada pasta, confirmado antes da movimentação.

---

## 2. Estado Atual por Especialista (Visão) — números corrigidos

| Cenário | Pasta origem | Imagens | Labels | Caixas (bboxes) | Média caixas/imagem | Vídeos-fonte identificados |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| **Frente Dia** | `ds_front_dia` | 2.022 | 2.022 | 16.185 | 8,01 | 8 |
| **Frente Noite** | `ds_front_noite` | 2.001 | 2.001 | 16.008 | 8,00 | 13 |
| **Cima Dia** | `ds_sup_dia` | 2.549 | 2.549 | 18.258 | 7,16 | 6 |
| **Cima Noite** | `ds_sup_noite` | 2.032 | 2.032 | 16.254 | 8,00 | 9 |
| **Total** | — | **8.604** | **8.604** | **66.705** | — | 36 |

(Antes da atualização: 5.204 imagens / 39.504 caixas / 11 vídeos-fonte no total. Crescimento de **+65% em imagens** nesta rodada, concentrado em `front_dia`, `front_noite` e `sup_noite`.)

Checagens de sanidade em todas as 4 pastas, pós-correção:
- 0 arquivos de label vazios.
- 0 divergência de nomes entre `images/` e `labels/`.
- Classe única `0` (`goat`) em todos os labels — consistente com `nc: 1, names: ['goat']`.

### Destaque: `front_noite` deixou de ser o especialista de maior risco de vazamento de cena
O CLAUDE.md documentava `front_noite` como tendo **apenas 1 vídeo-fonte** — o pior caso para vazamento de cena/sessão entre treino e validação (ver seção 3.2 do CLAUDE.md). Com a atualização, `front_noite` passou a ter **13 vídeos-fonte independentes** (gravações em datas/horários diferentes), a maior diversidade de fontes entre os 4 especialistas. Isso reduz substancialmente esse risco específico.

### Vídeos-fonte por especialista (pós-atualização)

- **Frente Dia** (8 vídeos): 2 vídeos antigos (`Cam01`, 451+721 frames) + 6 vídeos novos `Cam03` diurnos (100–150 frames cada).
- **Frente Noite** (13 vídeos): 1 vídeo antigo (`Cam03_..._redimencionado`, 451 frames) + 12 vídeos novos `Cam03` noturnos (50–150 frames cada).
- **Cima Dia** (6 vídeos, sem alteração nesta rodada): `Cam04` (1.670), `sequence_frames` (719, sem prefixo identificável), 4 blocos curtos `Cam02_...avi` (40 cada).
- **Cima Noite** (9 vídeos): 2 vídeos antigos (`Cam02`, 992+40 frames) + 7 vídeos novos `Cam04` noturnos (125–150 frames cada).

Nenhuma imagem foi excluída por nome fora do padrão regex em nenhum dos 4 especialistas (0 exclusões em todos, incluindo os novos arquivos).

---

## 3. Preparativos já realizados para o novo treinamento

1. **Arquivamento dos resultados do treino anterior** (para não perder as métricas atuais da tabela do TCC, já que o próximo treino sobrescreve `fold_1..fold_5` via `exist_ok=True`):
   - `yolo-8-*/runs/detect/` → renomeado para `yolo-8-*/runs/detect_archive_20260901/` (todos os 4 especialistas, ~530MB cada).
   - `yolo-8-*/runs/detect/` recriado vazio, pronto para os novos treinos.
   - `kfold_audit_report.txt` e `fold-metrics.txt` de cada especialista copiados para `*_archive_20260901.txt` antes da regeneração (esses arquivos de arquivo refletem o dataset **anterior** à correção do bug, isto é, ainda mais antigo que os números da seção 2).

2. **Correção do bug de detecção de imagens** (seção 1) aplicada em `generate_temporal_kfold.py` e nos dados em disco (achatamento das pastas aninhadas).

3. **Regeneração dos 5 folds temporais**, agora com o dataset completo e corrigido, via `python generate_temporal_kfold.py --purge-frames 15` (mesmo purge buffer usado anteriormente). Vazamento exato treino/validação: **0%** confirmado em todos os folds de todos os especialistas.

   Tamanhos resultantes por fold (treino / validação, após purge):

   | Especialista | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
   | :--- | :--- | :--- | :--- | :--- | :--- |
   | Frente Dia | 1496/406 | 1378/404 | 1378/404 | 1378/404 | 1498/404 |
   | Frente Noite | 1405/401 | 1219/400 | 1211/400 | 1219/400 | 1406/400 |
   | Cima Dia | 1949/510 | 1887/510 | 1859/510 | 1887/510 | 1950/509 |
   | Cima Noite | 1490/407 | 1363/406 | 1355/407 | 1363/406 | 1491/406 |

4. **Verificações de ambiente para o treino:**
   - `weights/yolov8m.pt` presente (pesos base pré-treinados, 52MB).
   - Cache antigo (`*.cache`) em `ds_*` é limpo automaticamente pelo próprio `run_parallel_kfold_trainings.py` (`clean_corrupted_caches()`) antes de iniciar — não precisa limpar manualmente.
   - Espaço em disco: 47GB livres em `/` e 57GB livres no HD externo — suficiente para os novos runs.

---

## 4. Como rodar o novo treinamento completo

Está tudo pronto. Basta rodar (ambiente `tcc` ativado ou usando o interpretador direto):

```bash
cd /home/vini/tcc/moe
python run_parallel_kfold_trainings.py --epochs 300
```

Isso vai treinar sequencialmente os 20 folds (4 especialistas × 5 folds), salvando em `yolo-8-*/runs/detect/fold_{1..5}/`. Com quase 2x mais dados em 3 dos 4 especialistas, espere tempo de treino proporcionalmente maior que a rodada anterior.

## 5. Passos manuais após o treino terminar (não incluídos neste levantamento)

1. `python extract_best_metrics.py` — recalcula e sobrescreve `tabela_resultados_yolo.{md,csv,tex}` com os melhores folds do **novo** treino.
2. Copiar manualmente os `best.pt` dos folds vencedores de cada especialista para `modelo-final/best-weights/best-<cenario>.pt`, substituindo os pesos consolidados atuais.
3. Atualizar a tabela da seção 5 e a observação sobre `front-noite` (1 vídeo-fonte) na seção 3.2 do `CLAUDE.md` com os novos números (ver seção 2 deste documento).
4. Os resultados antigos ficam preservados em `yolo-8-*/runs/detect_archive_20260901/` e nos `*_archive_20260901.txt` caso seja necessário comparar antes/depois no texto do TCC.
