# ENTENDIMENTO — Redes neurais do trabalho de PAI

Código: `trabalho/redesNeurais/neuralNetwork.py` (um arquivo só, dividido em 7 etapas).
Este documento explica **o que** foi feito, **por quê** e **cada função**, na ordem do código.

---

## 0. Como rodar

1. Dataset descompactado em `Base de dados/BreastDCEDL_ISPY1_min_crop/` (pastas `dce/` e `mask/`)
   e o CSV em `Base de dados/BreastDCEDL_SPY1_min_crop_metadata.csv`. A pasta `Base de dados/` fica
   na raiz do repositório e não vai para o git.
2. Ambiente: Python 3.12 + `torch torchvision nibabel scipy numpy pandas scikit-learn matplotlib`.
3. Rodar tudo, ou só uma parte:
   ```
   python trabalho/redesNeurais/neuralNetwork.py
   python trabalho/redesNeurais/neuralNetwork.py --redes resnet50 --regimes 3fases --estrategias central
   ```
4. Saídas:
   - `pesos/<rede>_<regime>_<estrategia>.pt`: pesos da melhor época (não vão para o git);
   - `resultados/metricas.csv`: uma linha por experimento;
   - `resultados/loss_*.png`: curvas de convergência;
   - `resultados/gradcam_*.png`: exemplos de Grad-CAM.

---

## 1. O que descobrimos sobre o dataset

- **172 pacientes**. Cada um tem 3 arquivos de fase em `dce/` e 1 máscara em `mask/`.
- Nome da fase: `{pid}_spy1_vis1_dce_aqc_{i}.nii.gz`. O `i` vem das colunas `pre`, `post_early` e
  `post_late` do CSV. Em 140 pacientes é `0/1/2`, mas em 31 é `0/2/3` ou `0/2/5`, por isso **o índice sempre
  vem do CSV**.
- Shape de cada volume: **(N, 256, 256)**. **O eixo 0 é o dos cortes** (N vai de 10 a 72), e cada corte já é 256×256.
  É o mesmo eixo que o `extrairCorteCentral.py` do Diego usa (lá chamado de "sagital").
- O volume já vem **recortado em volta do tumor** ("MinCrop"): há tumor do corte 2 até ~N−3.
  O corte do meio (`N // 2`) cai no tumor. Os campos `n_z`, `mask_start` e `mask_end` do CSV
  se referem ao exame **original** e não servem para o volume recortado.
- Intensidades **sem escala fixa**: o máximo é 110 em um paciente e 2.065 em outro → precisa normalizar.
- Coluna `test` do CSV (artigo do dataset, Tabela 4): **0 = treino, 1 = teste, 2 = validação**.

| Partição | Pacientes | sem pCR | com pCR |
|---|---|---|---|
| treino (`test=0`) | 104 | 78 | 26 |
| validação (`test=2`) | 33 | 22 | 11 |
| teste (`test=1`) | 35 | 23 | 12 |

---

## 2. As etapas, em tópicos

**Etapa 1 — Leitura:** lê as 3 fases de cada paciente → normaliza cada volume → escolhe os cortes
(o do meio, ou todos os que têm tumor) → empilha as 3 fases como 3 canais: `(3, 256, 256)`.

**Etapa 2 — Aumento de dados + Dataset:** as 10 variações do enunciado (5 ângulos × original/espelhada).
- Treino: sorteia 1 variação por imagem a cada época.
- Validação: usa as 10 variações.
- Teste: sem aumento.

**Etapa 3 — Redes:** CNN própria (6 camadas com parâmetros), ResNet50 congelada (treina só a FC)
e VGG16 congelada (treina só uma cabeça pequena). No regime de 1 fase, as redes do ImageNet recebem o canal repetido 3×.

**Etapa 4 — Treino:** loss BCE com **peso 3 para a classe pCR** (desbalanceamento), otimizador Adam,
early stopping pela loss de validação e salvamento dos pesos da melhor época.

**Etapa 5 — Avaliação:** probabilidade ≥ 0,5 → pCR. Métricas: sensibilidade, especificidade, precisão, acurácia, F1,
tempo de treino e curva de loss. Na estratégia "todos os cortes", junta os cortes por paciente (média ou máximo).

**Etapa 6 — Grad-CAM:** mapa de calor de onde a rede olhou, ao lado da máscara do tumor.

**Etapa 7 — Experimentos:** 3 redes × 2 regimes × 2 estratégias = 12 treinos. Tudo vai para `resultados/`.

---

## 3. Função por função

### Configurações (topo do arquivo)

| Nome | O que é |
|---|---|
| `PASTA_RAIZ`, `PASTA_BASE`, ... | caminhos montados a partir do próprio arquivo. Funcionam no Mac e no Windows sem editar |
| `FASES` | colunas do CSV com o número do arquivo de cada fase |
| `PARTICAO_TREINO/TESTE/VALIDACAO` | 0, 1, 2: significado da coluna `test` |
| `REGIMES` | `"1fase"` usa só o canal 1 (pós-contraste precoce); `"3fases"` usa os 3 |
| `ANGULOS`, `VARIACOES` | as 10 combinações (ângulo, espelhar) do aumento de dados |
| `SEMENTE`, `TAMANHO_LOTE`, `TAXA_APRENDIZADO`, ... | hiperparâmetros (justificativas na seção 4) |
| `dispositivo` | MPS (Mac), CUDA (NVIDIA) ou CPU, o que estiver disponível |

### `fixar_semente()`
Fixa os sorteios do `random`, do `numpy` e do `torch`. Assim, rodar de novo dá o mesmo resultado, e
a comparação entre 1 fase e 3 fases é justa (mesma inicialização, mesma ordem dos lotes).

### Etapa 1 — Leitura

**`ler_volume(caminho)`**: abre o `.nii.gz` com `nibabel` e devolve o array 3D em `float32`
(metade da memória do `float64` e o tipo que o PyTorch usa).

**`normalizar(volume)`**: z-score: `(volume − média) / desvio`. Cada volume passa a ter média 0 e desvio 1.
O `+ 1e-8` evita divisão por zero. A normalização é **por volume** porque cada exame tem sua escala.

**`ler_fases_do_paciente(linha)`**: para cada fase, pega no CSV o índice do arquivo, monta o nome,
lê e normaliza. Devolve `(3, N, 256, 256)`: fase × corte × altura × largura.

**`ler_mascara(pid)`**: lê a máscara do tumor (1 = tumor, 0 = fundo). É usada para escolher os cortes com
tumor e no Grad-CAM.

**`carregar_dados(estrategia)`**: percorre os 172 pacientes do CSV:
- `"central"`: pega 1 corte, `N // 2`;
- `"todos"`: pega todos os cortes `k` com `mascara[k].sum() > 0` (com tumor).

Para cada corte escolhido, guarda a imagem `(3, 256, 256)` em `float16`, o rótulo `pCR`, a partição e o `pid`.
O `pid` serve para juntar depois as previsões de cada paciente.
Devolve um dicionário de arrays: `imagens`, `rotulos`, `particoes` e `pacientes`.

### Etapa 2 — Aumento de dados e Dataset

**`aplicar_variacao(imagem, angulo, espelhar)`**: espelha (inverte as colunas) e gira com
`scipy.ndimage.rotate`, com os mesmos parâmetros do `aumentoDeDados.py` do grupo (`reshape=False`, `mode="nearest"`).
Diferenças em relação ao arquivo do grupo:
- aceita **vários canais** e gira as 3 fases juntas (`axes=(2, 1)`), para elas continuarem alinhadas;
- gera **uma** variação por chamada, em vez das 10.

**`class DatasetPCR`**: a "ponte" entre os arrays e o PyTorch. O `DataLoader` pede itens por posição e monta os lotes.
- `__init__`: guarda as imagens, os rótulos, os canais do regime e o modo de aumento.
- `__len__`: quantos itens existem. No modo `"todas"` é 10× o número de imagens.
- `__getitem__(posicao)`: escolhe a variação conforme o modo, recorta os canais do regime, converte para `float32`,
  aplica a variação e devolve `(tensor imagem, tensor rótulo)`.
  - `"sorteio"` (treino): `random.choice(VARIACOES)`;
  - `"todas"` (validação): `divmod(posicao, 10)` → (qual imagem, qual variação);
  - `"nenhum"` (teste): a imagem original.

### Etapa 3 — Redes

**`bloco_convolucional(entrada, saida)`**: conv 3×3 → BatchNorm → ReLU → MaxPool 2×2.
- A conv detecta padrões;
- o BatchNorm estabiliza o treino;
- a ReLU zera os negativos (não linearidade);
- o MaxPool reduz a imagem pela metade.

**`class CNNPropria`**: 4 blocos convolucionais (16 → 32 → 64 → 128 filtros) + média global + 2 FC.

| # | Camada com parâmetros | Saída |
|---|---|---|
| 1 | Conv 3×3, 16 filtros | 16×128×128 (após o pool) |
| 2 | Conv 3×3, 32 filtros | 32×64×64 |
| 3 | Conv 3×3, 64 filtros | 64×32×32 |
| 4 | Conv 3×3, 128 filtros | 128×16×16 |
| — | média global (sem parâmetros) | 128 |
| 5 | FC 128 → 64 + ReLU + Dropout 0,5 | 64 |
| 6 | FC 64 → 1 | 1 (logit) |

- ~106 mil parâmetros, **todos treinados**.
- `forward`: extrator → média global → classificador.
- `camada_gradcam`: a ReLU do último bloco convolucional.

**`repetir_para_3_canais(x)`**: se a entrada tem 1 canal, repete 3 vezes. As redes do ImageNet foram
treinadas com fotos RGB e a primeira camada espera 3 canais. Repetir mantém os pesos pré-treinados.

**`class ResNet50Congelada`**:
- carrega a ResNet50 com os pesos do ImageNet (`IMAGENET1K_V2`);
- congela todos os parâmetros (`requires_grad = False`);
- troca a `fc` (2048 → 1000) por `Dropout(0,5) + Linear(2048 → 1)`. Só ela treina: 2.049 parâmetros de 23,5 milhões.
- `train(mode)`: sobrescrita para a base **ficar sempre em modo de avaliação**. Sem isso, o BatchNorm da base
  recalcularia as médias com os nossos lotes, mesmo congelado, e estragaria o que foi aprendido no ImageNet.
- `forward`: repete o canal, se preciso, e passa pela rede.
- `camada_gradcam`: `layer4`, o último estágio convolucional (2048 mapas de 8×8 para a entrada de 256).

**`class VGG16Congelada`**:
- carrega a VGG16 com os pesos do ImageNet e congela tudo;
- troca o `avgpool` por média global (512 valores) e o `classifier` (~120 milhões de parâmetros) por
  `FC 512 → 64 + ReLU + Dropout 0,5 + FC 64 → 1` (~33 mil parâmetros). Treinar 120 milhões com 104 pacientes seria decorar;
- deixa as ReLU com `inplace=False`, o que é necessário para o Grad-CAM ler as ativações;
- `train(mode)`: mantém as convoluções (`features`) em modo de avaliação;
- `camada_gradcam`: `features[29]`, a ReLU da última convolução.

**`REDES`**: dicionário nome → classe (`"cnn"`, `"resnet50"`, `"vgg16"`), para escolher a rede pelo nome.

**`contar_parametros(modelo)`**: devolve (treináveis, total). Vai para o CSV de métricas e para o relatório.

### Etapa 4 — Treino

**`calcular_peso_positivo(rotulos_treino)`**: `nº sem pCR / nº com pCR` no **treino** = 78/26 = **3**.
É o tratamento do desbalanceamento: errar um paciente com pCR pesa 3× na loss.

**`executar_epoca(modelo, carregador, funcao_perda, otimizador=None)`**: uma passada por todos os lotes.
- Com otimizador (treino), em cada lote: forward → loss → `zero_grad` → `backward` → `step`.
- Sem otimizador (validação): só forward e loss, sem calcular gradiente (`set_grad_enabled(False)`).
- Devolve a loss média da época.

**`treinar(...)`**:
1. cria a loss `BCEWithLogitsLoss(pos_weight=3)` e o otimizador Adam, **só com os parâmetros treináveis**;
2. a cada época treina, valida e guarda as duas losses;
3. se a loss de validação melhorou, salva os pesos (`torch.save`) e zera o contador;
4. se ficar `PACIENCIA` épocas sem melhorar, para (**early stopping**);
5. no fim, recarrega os pesos da melhor época e devolve o histórico (losses, melhor época, tempo).

### Etapa 5 — Avaliação

**`prever(modelo, carregador)`**: modo de avaliação, sem gradiente. Aplica a sigmoide nos logits e devolve
as probabilidades de pCR.

**`agregar_por_paciente(probabilidades, rotulos, pacientes, modo)`**: estratégia "todos os cortes".
Agrupa as probabilidades pelo `pid` e tira a **média** ou o **máximo**. Devolve uma probabilidade e um rótulo por paciente.

**`calcular_metricas(rotulos, probabilidades, limiar)`**: probabilidade ≥ 0,5 → pCR. Monta a matriz de
confusão (VP, VN, FP, FN) e calcula:
- sensibilidade = VP/(VP+FN);
- especificidade = VN/(VN+FP);
- precisão = VP/(VP+FP);
- acurácia = (VP+VN)/total;
- F1 = 2·P·S/(P+S).

A função `dividir` devolve 0 quando o denominador é 0, por exemplo se a rede nunca prever pCR.

**`salvar_curva_perda(historico, titulo, caminho)`**: gráfico da loss de treino × validação por época,
com uma linha tracejada na melhor época.

### Etapa 6 — Grad-CAM

**`gerar_gradcam(modelo, imagem)`**:
1. um *hook* (gancho) na camada escolhida guarda as **ativações** no forward e registra outro gancho que guarda o
   **gradiente** delas no backward;
2. `requires_grad_(True)` na entrada faz o gradiente atravessar a base congelada;
3. forward → backward da saída (o logit de pCR);
4. peso de cada mapa = média do gradiente dele;
5. soma ponderada dos mapas → ReLU (só o que **aumenta** a chance de pCR);
6. amplia para 256×256 e normaliza entre 0 e 1.

Devolve (probabilidade, mapa).

**`salvar_figura_gradcam(...)`**: três painéis: a imagem (pós-contraste precoce), o Grad-CAM sobreposto e a máscara
do tumor. Assim dá para ver se a rede olhou para o tumor (pergunta do relatório).

### Etapa 7 — Experimentos

**`separar(dados, particao)`**: filtra todos os arrays de uma partição.

**`executar_experimento(nome_rede, regime, estrategia, dados)`**: um experimento completo:
1. fixa a semente e separa treino, validação e teste;
2. cria os 3 `DataLoader`. O de treino é embaralhado e com sorteio de variação. O de validação usa as 10 variações
   na estratégia central e a imagem original em "todos" (validar 10× todos os cortes multiplicaria o tempo por 10).
   O de teste usa a imagem original;
3. cria a rede com 1 ou 3 canais;
4. treina e salva os pesos e a curva de loss;
5. prevê o teste. Na estratégia central, cada imagem é um paciente; em "todos", agrega por média e por máximo;
6. devolve as linhas de métricas e o modelo treinado.

**`salvar_metricas(linhas, caminho)`**: acrescenta as linhas no `resultados/metricas.csv`.

**`gerar_exemplo_gradcam(modelo, nome, regime, dados)`**: pega o primeiro paciente com pCR do teste e salva a
figura de Grad-CAM.

**`main()`**: lê as opções (`--redes`, `--regimes`, `--estrategias`), cria as pastas e, para cada estratégia,
carrega os dados uma vez e roda todas as combinações de rede × regime.

---

## 4. Hiperparâmetros e por quê

| Hiperparâmetro | Valor | Por quê |
|---|---|---|
| Fase do regime "1 fase" | pós-contraste precoce | é a fase em que o tumor mais realça (pico do contraste) |
| Normalização | z-score por volume | cada exame tem escala própria; média 0 e desvio 1 também é a escala que as redes do ImageNet esperam |
| Tamanho da imagem | 256×256 (sem redimensionar) | ResNet e VGG terminam em pooling adaptativo, então aceitam 256 |
| Tamanho do lote | 16 | o treino tem 104 imagens (≈ 7 lotes por época); lotes pequenos ajudam a generalizar e cabem na memória |
| Otimizador | Adam | ajusta o passo de cada peso sozinho; converge rápido com pouco ajuste |
| Taxa de aprendizado | 1e-3 | padrão do Adam; como as bases estão congeladas, só as cabeças (novas) aprendem, e elas toleram LR maior |
| Weight decay | 1e-4 | regularização leve contra overfitting |
| Dropout | 0,5 | valor clássico; reduz a dependência de poucos neurônios |
| Loss | BCE com `pos_weight = 3` | problema binário + desbalanceamento 78:26 no treino |
| Limiar | 0,5 | padrão; pode ser ajustado na validação (próximo passo) |
| Épocas máx. / paciência | central: 40 / 8; todos: 15 / 4 | "todos" tem ~27× mais imagens por época, então converge em menos épocas |
| Congelamento | ResNet50: só a FC treina; VGG16: só a cabeça | o enunciado pede isso na ResNet50; com 104 pacientes, treinar milhões de pesos decoraria o treino |
| Cabeça da VGG16 | média global + 512→64→1 | a cabeça original (~120 milhões) seria grande demais |
| CNN própria | 4 conv + 2 FC, filtros 16→128 | ≥ 5 camadas com parâmetros (exigência), pequena (~106 mil) pelo pouco dado; filtros 3×3 como na VGG; média global no lugar de uma FC grande |
| Semente | 42 | reprodutibilidade |

---

## 5. Resultados

Os resultados ficam em `resultados/metricas.csv` e nos gráficos de `resultados/`. Um resumo vai na seção abaixo.

### Primeira rodada (30/10, hiperparâmetros da seção 4, sem ajuste fino)

Conjunto de **teste** (35 pacientes: 23 sem pCR, 12 com pCR). Rodado no Mac M1 (MPS).

| Rede | Regime | Estratégia | Sens. | Espec. | Prec. | Acur. | F1 | Tempo (s) |
|---|---|---|---|---|---|---|---|---|
| CNN | 1 fase | central | 0,58 | 0,52 | 0,39 | 0,54 | 0,47 | 47 |
| CNN | 3 fases | central | 1,00 | 0,04 | 0,35 | 0,37 | 0,52 | 30 |
| ResNet50 | 1 fase | central | 0,75 | 0,30 | 0,36 | 0,46 | 0,49 | 76 |
| ResNet50 | 3 fases | central | 0,67 | 0,39 | 0,36 | 0,49 | 0,47 | 85 |
| VGG16 | 1 fase | central | 0,92 | 0,09 | 0,34 | 0,37 | 0,50 | 126 |
| VGG16 | 3 fases | central | 0,58 | 0,35 | 0,32 | 0,43 | 0,41 | 141 |
| CNN | 1 fase | todos (média) | 0,83 | 0,26 | 0,37 | 0,46 | 0,51 | 182 |
| CNN | 1 fase | todos (máximo) | 0,92 | 0,09 | 0,34 | 0,37 | 0,50 | 182 |
| CNN | 3 fases | todos (média) | 0,50 | 0,39 | 0,30 | 0,43 | 0,38 | 143 |
| CNN | 3 fases | todos (máximo) | 0,83 | 0,26 | 0,37 | 0,46 | 0,51 | 143 |
| ResNet50 | 1 fase | todos (média) | 0,83 | 0,35 | 0,40 | 0,51 | **0,54** | 390 |
| ResNet50 | 1 fase | todos (máximo) | 1,00 | 0,00 | 0,34 | 0,34 | 0,51 | 390 |
| ResNet50 | 3 fases | todos (média) | 0,58 | 0,65 | 0,47 | 0,63 | 0,52 | 300 |
| ResNet50 | 3 fases | todos (máximo) | 0,92 | 0,00 | 0,32 | 0,31 | 0,48 | 300 |
| VGG16 | 1 fase | todos (média) | 0,33 | 0,87 | 0,57 | **0,69** | 0,42 | 404 |
| VGG16 | 1 fase | todos (máximo) | 0,67 | 0,48 | 0,40 | 0,54 | 0,50 | 404 |
| VGG16 | 3 fases | todos (média) | 0,17 | 0,78 | 0,29 | 0,57 | 0,21 | 443 |
| VGG16 | 3 fases | todos (máximo) | 0,67 | 0,52 | 0,42 | 0,57 | 0,52 | 443 |

**Leitura honesta:**
- **Todo o pipeline funciona** de ponta a ponta: 12 modelos treinados, pesos salvos, curvas de loss e Grad-CAM gerados.
- **O desempenho ainda está perto do acaso.** É esperado nesta primeira rodada: são só 104 pacientes de treino, e prever pCR
  só pela imagem é difícil até em artigos.
- **A melhor época costuma ser a 1–4** e depois a loss de validação sobe: as redes decoram o treino rápido (overfitting).
  O primeiro ajuste a testar é LR menor (1e-4) e mais regularização.
- **"Máximo dos cortes" quase sempre diz "pCR"** (sensibilidade alta e especificidade ~0): basta um corte acima de 0,5.
  Com esse modo de agregação, o limiar precisa ser maior.
- **Várias redes estão "presas" em uma classe** (sensibilidade 1,0 e especificidade ~0, ou o contrário). Ajustar o
  **limiar na validação** deve equilibrar.
- **Ainda não dá para afirmar que 3 fases > 1 fase**: a diferença está dentro do ruído de 35 pacientes de teste.

### Próximos ajustes (para a tabela de tentativas do relatório)
1. LR 1e-4 nas cabeças e na CNN, e comparar com 1e-3.
2. Escolher o limiar que maximiza o F1 na **validação** e aplicá-lo no teste.
3. VGG16: testar fine-tuning do último bloco (LR 1e-5).
4. Opcional: máscara como 4º canal ou fundo zerado.

---

## 6. Pontos para o grupo

- **Para o Diego:** `carregar_dados()` já faz o que o `extrairDatasets.py` pretendia: liga cada imagem ao CSV
  (rótulo, partição e fase certa), normaliza e usa as 3 fases. O eixo que ele escolheu está certo.
  Se ele preferir manter as funções dele, basta que devolvam o mesmo formato:
  `imagens (N, 3, 256, 256)`, `rotulos (N,)`, `particoes (N,)` e `pacientes (N,)`.
- **Aumento de dados:** a versão aqui gira as 3 fases juntas e gera uma variação por vez. O
  `aumentoDeDados.py` continua útil para mostrar as 10 variações na interface.
- **Entrega final é um `.py` único:** este arquivo já está organizado por etapas para facilitar a junção com a
  interface (`interface.py` / `visualizador.py`).
- **Ainda falta:**
  - ligar a classificação e o Grad-CAM à interface (requisito e);
  - mostrar a máscara ao lado no visualizador (requisito a);
  - ajustar o limiar e os hiperparâmetros com base na validação e registrar as tentativas (o relatório pede isso);
  - opcional: máscara como canal extra.
