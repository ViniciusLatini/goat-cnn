# Tutorial de Uso - Pipeline de Predição MoE (CNN + YOLO)

Este diretório contém o pipeline Mixture of Experts (MoE) para detecção de cabras. O script classifica o cenário (luz e ângulo) usando uma CNN e ativa o modelo YOLOv8 especialista ideal.

## Pré-requisitos

O pipeline necessita do ambiente virtual Conda `tcc` configurado no sistema.

Para utilizar o ambiente correto, execute os comandos especificando o interpretador Python do ambiente `tcc` (`/home/vini/miniconda3/envs/tcc/bin/python`) ou ative-o antes de executar:

```bash
conda activate tcc
```

As bibliotecas necessárias instaladas no ambiente são:
- `torch` & `torchvision` (PyTorch)
- `ultralytics` (YOLOv8)
- `opencv-python` (`cv2`)
- `pillow` (`PIL`)

---

## Como Usar o Script

O script principal é o `predicao_moe.py`. Ele aceita argumentos por linha de comando para definir o arquivo de entrada (vídeo ou imagem) e a pasta de saída.

### Comando Básico
```bash
python predicao_moe.py --input <caminho_da_midia> --output <pasta_de_saida>
```

Ou usando o caminho absoluto do Python caso o ambiente conda não esteja ativado:
```bash
/home/vini/miniconda3/envs/tcc/bin/python predicao_moe.py -i <caminho_da_midia> -o <pasta_de_saida>
```

### Argumentos:
* `-i`, `--input` (obrigatório): Caminho para a imagem ou vídeo a ser processado.
* `-o`, `--output` (opcional, padrão: `resultados`): Caminho para a pasta onde os resultados finais de predição e os logs serão salvos.

---

## Exemplos Práticos

### 1. Predição em uma Imagem
```bash
python predicao_moe.py -i ../goat-cnn/test_imgs/test1-001.jpg -o resultados
```
**O que acontece:**
1. A CNN classifica a imagem e identifica o cenário (ex: `Frente Dia`).
2. O script carrega o modelo YOLO especialista `best-front-dia.pt`.
3. Executa a predição e desenha as caixas delimitadoras nos objetos detectados.
4. Salva a imagem anotada em `resultados/predicao_moe/test1-001.jpg`.
5. Salva um resumo textual detalhado das detecções em `resultados/predicao_moe/resumo.txt`.

### 2. Predição em um Vídeo
```bash
python predicao_moe.py -i ../goat-cnn/videos/class_1/Cam01_2025_04_25_16_25_04_capril.avi -o resultados
```
**O que acontece:**
1. O script extrai a primeira frame do vídeo.
2. A CNN classifica o frame para determinar o cenário (ex: `Frente Dia`).
3. O script carrega o modelo YOLO especialista correspondente.
4. O YOLO processa o vídeo inteiro e salva o vídeo anotado final com as caixas de detecção em `resultados/predicao_moe/Cam01_2025_04_25_16_25_04_capril.avi`.
5. Salva o resumo de execução em `resultados/predicao_moe/resumo.txt`.

---

## Estrutura de Arquivos e Especialistas

O script faz o seguinte mapeamento automático com base na saída do classificador EfficientNet-B0:

| Classe CNN | Descrição do Cenário | Modelo Especialista Carregado |
| :--- | :--- | :--- |
| `class_1` | Frente Dia | `best-front-dia.pt` |
| `class_2` | Frente Noite | `best-front-noite.pt` |
| `class_3` | Cima Dia | `best-sup-dia.pt` |
| `class_4` | Cima Noite | `best-sup-noite.pt` |
