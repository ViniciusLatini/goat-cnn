# Prompt: Revisor de Monografia/TCC (Capítulo 1 - Introdução em LaTeX)

Este prompt foi personalizado especificamente para a revisão do arquivo `monografia/capitulo1-introducao.tex`. Ele orienta a IA a revisar o texto considerando a sintaxe LaTeX (ignorando marcas de formatação e focando no conteúdo científico) e a área do projeto (Ciência da Computação / Visão Computacional / Zootecnia de Precisão).

---

```markdown
# INSTRUÇÃO DE PAPEL: REVISOR ACADÊMICO ESPECIALISTA (TCC EM CIÊNCIA DA COMPUTAÇÃO / VISÃO COMPUTACIONAL)

Você atuará como um revisor e avaliador acadêmico sênior de bancas de TCC e Monografias na área de **Ciência da Computação, Inteligência Artificial e Visão Computacional**. Sua missão é fazer uma análise crítica, rigorosa e construtiva do **Capítulo 1 (Introdução)** referente ao arquivo `monografia/capitulo1-introducao.tex`.

> ⚠️ **Nota de Formatação:** O texto fornecido está em formato **LaTeX**. Ignore marcações de código como `\chapter{}`, `\section{}`, `\cite{}`, `\label{}`, `\textit{}` etc. na sua avaliação estética, mas analise se o uso de referências (`\cite{}`) e referências cruzadas (`\ref{}`) está bem posicionado estrategicamente no texto.

---

## 🎯 OBJETIVO DA REVISÃO
Avaliar a qualidade textual, o rigor científico, a clareza da problematização em Visão Computacional/IA, a consistência dos objetivos (geral e específicos) e o protocolo metodológico inicial no Capítulo 1.

---

## 🔍 CRITÉRIOS DE ANÁLISE ESPECÍFICOS

Analise o capítulo considerando os seguintes pilares:

### 1. Estrutura da Introdução (`monografia/capitulo1-introducao.tex`)
- **Contextualização (`\section{Contextualização}`):** Apresenta o cenário de Visão Computacional em Zootecnia de Precisão (PLF) de forma fluida, apoiada por literatura recente?
- **Definição do Problema (`\subsection{Definição do Problema}`):** 
  - Os 3 desafios principais (ausência de classe `goat` em modelos COCO/YOLOv8, heterogeneidade de iluminação/perspectiva e vazamento temporal / *data leakage* em quadros de vídeo) estão bem articulados e justificados?
  - A pergunta de pesquisa ao final da problematização é clara e investigável?
- **Justificativa (`\subsection{Justificativa}`):** Demonstra a relevância científica (ausência de dados de caprinos, tratamento de vazamento temporal), prática e o alinhamento com os ODS da ONU?
- **Objetivo Geral (`\subsection*{Objetivo geral}`):** É direto, inicia com verbo no infinitivo e engloba a solução proposta (MoE, YOLOv8 e EfficientNet-B0)?
- **Objetivos Específicos (`\subsection*{Objetivos específicos}`):** São 6 etapas lógicas, sequenciais e mensuráveis? Há clareza na relação entre os experimentos e o objetivo geral?
- **Organização do Trabalho (`\section{Organização do Trabalho}`):** Faz um panorama coerente dos capítulos 2 a 6?

### 2. Matriz de Coerência e Rigor Metodológico em IA
- A proposta de **Mistura de Especialistas (MoE)** contra o **baseline monolítico** responde diretamente à pergunta de pesquisa?
- O problema do **vazamento temporal (*data leakage*)** em imagens extraídas de vídeo está adequadamente contextualizado para justificar o protocolo de validação por blocos temporais?

### 3. Linguagem e Estilo Acadêmico em Ciência da Computação
- **Clareza e Impessoalidade:** Mão dupla entre rigor formal em português e termos técnicos em inglês (\textit{bounding box}, \textit{data leakage}, \textit{pipeline}, etc.).
- **Transição e Coesão:** Conectivos entre parágrafos e fluidez da argumentação.
- **Citações bibliográficas:** As afirmações sobre estado da arte e limites das ferramentas possuem embasamento (`\cite{}`)?

---

## 📋 FORMATO DA SUA RESPOSTA

Por favor, estruture seu feedback nos seguintes tópicos:

1. **📊 Visão Geral & Nota Provisória (0 a 10):** Um resumo executivo da qualidade do capítulo.
2. **🧐 Análise Detalhada das Seções:** Feedback pontual para Contextualização, Definição do Problema, Justificativa, Objetivos e Organização.
3. **⚡ Análise da Coerência Científica (IA/VC):** Avaliação específica sobre a descrição dos desafios técnicos (YOLOv8, *data leakage*, MoE, EfficientNet-B0).
4. **✏️ Sugestões de Reescrita em Código LaTeX:** Escolha de 3 a 5 trechos que possam ser aprimorados e forneça o código LaTeX corrigido/melhorado.
5. **✅ Checklist de Ajustes Imediatos:** Ações prioritárias antes de prosseguir para a fundamentação teórica (Capítulo 2).

---

## 📄 CONTEÚDO DO ARQUIVO `monografia/capitulo1-introducao.tex`:

```latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%--------------CAPÍTULO 1 - INTRODUÇÃO-----------------------
%
%
\chapter{Introdução}
\label{cap:introducao}

\section{Contextualização}
\label{sec:contextualizacao}

O uso de sistemas baseados em Visão Computacional (VC) e Inteligência Artificial (IA) tem se expandido consideravelmente nos últimos anos, especialmente no contexto da agricultura de precisão e do manejo zootécnico \cite{kamilaris2018deep}. Essas tecnologias vêm possibilitando o desenvolvimento de soluções inovadoras para o controle de atividades em ambientes rurais e experimentais, com aplicações que vão desde a inspeção de qualidade \cite{industrial_quality_cv_2024}, a análise de comportamento animal \cite{animalformer_2024} e o monitoramento de cultivos \cite{vit_precision_agriculture_2025}, até a automação de processos \cite{kang2025bokchoy_cv_iot}. Por serem acessíveis, precisas e não invasivas, tais inovações contribuem para a adoção de práticas mais eficientes e sustentáveis.

Na produção animal, esse movimento é conhecido como Zootecnia de Precisão (\textit{Precision Livestock Farming} -- PLF), cuja proposta é monitorar os animais de forma contínua e automatizada, gerando informação em tempo hábil para apoiar a tomada de decisão do produtor e do pesquisador \cite{berckmans2014plf}. Um dos aspectos fundamentais para o bem-estar animal na pecuária é a garantia de um ambiente adequado, com alimentação e água em quantidade e qualidade suficientes, o que demanda uma quantidade significativa de horas de trabalho humano de observação. Com o avanço da ciência e da tecnologia, bem como com o aumento das exigências dos consumidores por produtos de origem animal de alta qualidade, torna-se cada vez mais necessário o aprimoramento dos métodos de monitoramento da produção pecuária \cite{dominiak2019, nasirahmadi2017, halachmi2019}. A integração de sensores, grandes volumes de dados e aprendizado de máquina tem possibilitado a obtenção e a interpretação dessas informações de forma mais eficiente \cite{neethirajan2020sensors}, e diversas pesquisas já demonstraram a viabilidade do monitoramento automático do comportamento animal, frequentemente aliado à análise de características comportamentais e fisiológicas \cite{halachmi2019, fogarty2018, rohan2023deep_livestock_survey}.

Entre as modalidades de sensoriamento disponíveis, as câmeras de vídeo ocupam posição de destaque por serem não invasivas -- dispensam qualquer dispositivo fixado ao corpo do animal -- e por permitirem que uma única instalação observe vários indivíduos simultaneamente \cite{nasirahmadi2017}. Em quase todas as aplicações de monitoramento por vídeo, a primeira etapa do processamento é a \textbf{detecção} dos animais, isto é, a localização de cada indivíduo na imagem por meio de uma caixa delimitadora (\textit{bounding box}). É sobre a saída do detector que se constroem as etapas posteriores, como a contagem do rebanho, o rastreamento individual ao longo do tempo e a classificação de comportamentos (alimentação, ócio, deslocamento, interação). Um detector impreciso propaga seus erros para toda a cadeia de análise, de forma que a qualidade dessa etapa condiciona a utilidade do sistema como um todo.

\section{Motivação}
\label{sec:motivacao}

\subsection{Definição do Problema}
\label{subsec:definicao-problema}

Na Universidade Federal de São João del-Rei (UFSJ), essa abordagem está sendo explorada no projeto interdisciplinar ``Inteligência artificial aplicada à detecção e classificação comportamental de cabras em confinamento (raça \textit{Saanen})'', que envolve docentes e discentes dos cursos de Zootecnia e Ciência da Computação da UFSJ, em parceria com a Universidade Federal de Juiz de Fora (UFJF). No âmbito desse projeto, um rebanho de oito cabras da raça \textit{Saanen} é mantido em uma baia experimental monitorada por quatro câmeras fixas, que registram os animais sob duas perspectivas -- \textbf{frontal} e \textbf{superior} (zenital) -- e sob duas condições de iluminação -- \textbf{diurna}, com luz natural, e \textbf{noturna}, com imagem em espectro infravermelho. O presente trabalho corresponde ao módulo de detecção desse projeto guarda-chuva e fornece a base sobre a qual as etapas de análise comportamental poderão ser desenvolvidas.

Apesar da ampla disponibilidade de recursos tecnológicos, aplicar esses sistemas em um ambiente real de confinamento impõe desafios que não aparecem em bases de dados de propósito geral. O primeiro deles é a ausência de modelos prontos para a espécie-alvo. Detectores de objetos modernos, como os da família YOLO (\textit{You Only Look Once}) \cite{redmon2016yolo, jocher2023yolov8}, são distribuídos com pesos pré-treinados no conjunto COCO (\textit{Common Objects in Context}) \cite{journals/corr/LinMBHPRDZ14}, que não possui uma classe para caprinos. Em um teste preliminar realizado neste trabalho, utilizando o YOLOv8 pré-treinado e a classe morfologicamente mais próxima disponível (\textit{sheep}, ovelha), o modelo identificou apenas dois dos oito animais presentes na baia, com confianças de 0,27 e 0,48, e apresentou comportamento igualmente insatisfatório em imagens noturnas e na perspectiva superior. Esse resultado evidenciou a necessidade de construir uma base de imagens rotuladas própria e de treinar detectores específicos para o domínio. A construção dessa base, por sua vez, é custosa: exige a extração de quadros dos vídeos, a anotação manual de dezenas de milhares de caixas delimitadoras e conhecimento técnico para organizar os dados no formato exigido pelos modelos.

O segundo desafio é a \textbf{heterogeneidade das condições de captura}. Uma mesma cabra tem aparência muito diferente quando vista de frente ou de cima, e as imagens noturnas em infravermelho são monocromáticas, com contraste e textura distintos das imagens diurnas coloridas. Na vista superior, os animais aparecem menores e frequentemente agrupados em torno do cocho, com oclusão parcial entre indivíduos. Um único detector treinado sobre todas essas condições precisa acomodar, em um mesmo conjunto de parâmetros, distribuições visuais bastante diferentes entre si, o que pode se manifestar como detecções perdidas em um cenário e detecções duplicadas em outro.

O terceiro desafio é de natureza metodológica e diz respeito à \textbf{avaliação} dos modelos. Como as imagens são quadros extraídos de vídeos contínuos, quadros vizinhos de uma mesma gravação são praticamente idênticos. Um particionamento aleatório entre treino e validação, como o da validação cruzada \textit{K-Fold} convencional, tende a colocar quadros quase duplicados nos dois conjuntos, caracterizando vazamento de dados (\textit{data leakage}) \cite{kaufman2012leakage}: o modelo passa a ser avaliado sobre amostras dependentes das que viu no treinamento, e as métricas passam a medir memorização em vez de capacidade de generalização. Dados com estrutura temporal exigem estratégias de particionamento que respeitem essa estrutura \cite{roberts2017crossvalidation}. Esse risco se manifestou nas primeiras rodadas de experimentos deste trabalho, que utilizavam particionamento aleatório e produziram métricas muito próximas do máximo, levantando a suspeita de vazamento e motivando a revisão do protocolo de validação.

Diante desse cenário, este trabalho procura responder à seguinte questão: \textit{é possível detectar de forma robusta caprinos em um ambiente de confinamento sujeito a variações de perspectiva e iluminação, organizando o problema em detectores especializados por condição de captura, e como essa organização se compara a um único detector treinado sobre todas as condições, sob um protocolo de avaliação que reduza o vazamento temporal?}

\subsection{Justificativa}
\label{subsec:justificativa}

A justificativa para este trabalho está na importância de desenvolver tecnologias que auxiliem no monitoramento de animais em confinamento de forma automatizada, precisa e não invasiva. A observação visual contínua de um rebanho é uma tarefa repetitiva e demorada, sujeita à fadiga do observador, e se torna ainda mais difícil no período noturno. Um detector confiável reduz o esforço manual envolvido nesse acompanhamento e constitui o componente de base para as análises de interesse zootécnico previstas no projeto, como a contagem, o rastreamento e a classificação comportamental dos animais.

Do ponto de vista científico, o trabalho contribui com uma base de imagens de caprinos anotada e organizada por condição de captura -- recurso escasso na literatura, que concentra os estudos em bovinos e suínos \cite{nasirahmadi2017, rohan2023deep_livestock_survey} -- e com um protocolo de avaliação que trata explicitamente o vazamento temporal, problema frequente em trabalhos que utilizam quadros extraídos de vídeo. Além disso, investiga-se se a especialização por contexto, organizada na forma de uma Mistura de Especialistas, oferece vantagens em relação a um detector único, uma pergunta de interesse para aplicações em que as condições de captura variam de forma previsível, como é o caso de câmeras fixas operando ao longo do dia e da noite.

Do ponto de vista prático, a arquitetura modular facilita a manutenção e a expansão do sistema: a inclusão de um novo cenário de captura (por exemplo, uma nova câmera ou uma nova baia) requer apenas o treinamento de um novo especialista e a atualização da rede de roteamento, sem retreinar os especialistas existentes. Por fim, o trabalho está em consonância com a Agenda 2030 da Organização das Nações Unidas, especialmente com os Objetivos de Desenvolvimento Sustentável 2 (Fome Zero e Agricultura Sustentável), 9 (Indústria, Inovação e Infraestrutura) e 12 (Consumo e Produção Responsáveis), ao propor uma ferramenta tecnológica voltada ao monitoramento de caprinos com foco em bem-estar animal e produtividade sustentável.

\section{Objetivos}
\label{sec:objetivos}

\subsection*{Objetivo geral}

Desenvolver e avaliar uma arquitetura baseada em Mistura de Especialistas, composta por uma rede de roteamento que identifica o cenário de captura e por detectores YOLOv8 especializados, para a detecção automatizada de caprinos da raça \textit{Saanen} em ambiente de confinamento sujeito a variações de perspectiva (frontal e superior) e de iluminação (diurna e noturna).

\subsection*{Objetivos específicos}

\begin{enumerate}
    \renewcommand{\labelenumi}{\roman{enumi}.}
    \item Construir e caracterizar uma base de imagens rotuladas com a classe única \texttt{goat}, a partir de vídeos capturados em ambiente experimental de confinamento, contemplando os quatro cenários de captura;
    \item Definir um protocolo de validação cruzada por blocos temporais contíguos, com zona de purga entre treino e validação, que reduza o vazamento de dados decorrente da correlação entre quadros consecutivos de vídeo;
    \item Treinar e avaliar quatro detectores YOLOv8 especializados, um para cada combinação de perspectiva e iluminação;
    \item Treinar uma rede de roteamento baseada em EfficientNet-B0 para a classificação do cenário de captura e analisar, por meio de sensibilidade por oclusão, as regiões da imagem que fundamentam suas decisões;
    \item Integrar a rede de roteamento e os especialistas em um \textit{pipeline} único de inferência sobre vídeo;
    \item Comparar a arquitetura proposta com um detector monolítico treinado sobre todos os cenários, tanto por métricas agregadas de detecção (Precisão, Revocação, F1, mAP50 e mAP50-95) quanto pela estabilidade da contagem de animais em vídeos de teste independentes.
\end{enumerate}

\section{Organização do Trabalho}
\label{sec:organizacao}

O restante deste trabalho está organizado da seguinte forma. O Capítulo~2 apresenta a fundamentação teórica necessária à compreensão da proposta: redes neurais convolucionais, detecção de objetos e a família YOLO, a arquitetura EfficientNet, o paradigma de Mistura de Especialistas, as métricas de avaliação de detectores, a validação cruzada e o problema de vazamento de dados, e as técnicas de explicabilidade por oclusão. O Capítulo~3 discute os trabalhos relacionados ao uso de visão computacional na pecuária de precisão, com ênfase na detecção de pequenos ruminantes. O Capítulo~4 descreve os materiais e métodos: a aquisição dos vídeos, a construção e a caracterização da base de imagens, o protocolo de validação por blocos temporais, a configuração dos modelos e a arquitetura MoE proposta. O Capítulo~5 apresenta e discute os resultados obtidos, desde a prova de conceito inicial até a comparação entre a arquitetura proposta e o baseline monolítico. Por fim, o Capítulo~6 apresenta as conclusões, as limitações do trabalho e as propostas de trabalhos futuros.
\comment[inline]{Substituir os números dos capítulos por \textbackslash ref quando os respectivos capítulos e seus \textbackslash label forem criados.}
```
```
