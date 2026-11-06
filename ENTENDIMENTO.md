# ENTENDIMENTO — o código do trabalho de PAI, função por função

Código: `trabalho_pai.py` (um arquivo só, dividido em 9 etapas).
Este documento explica **o que** foi feito, **por quê** e **cada função**, na ordem do código.
Serve para o grupo estudar: na apresentação, qualquer um pode ser chamado a explicar ou alterar qualquer parte.

---

## 0. Como rodar

1. Dataset descompactado em `Base de dados/BreastDCEDL_ISPY1_min_crop/` (pastas `dce/` e `mask/`).
   A pasta `Base de dados/` fica na raiz do repositório e não vai para o git.
   O CSV `BreastDCEDL_SPY1_min_crop_metadata.csv` já está na raiz do repositório
   (o programa também o procura em `Base de dados/`).
2. Ambiente: Python 3.12 +
   `torch torchvision nibabel scipy numpy pandas scikit-learn matplotlib customtkinter`.
3. Comandos:
   ```
   python trabalho_pai.py                 # abre a interface gráfica
   python trabalho_pai.py --treinar       # treino final: 3 redes x 2 regimes x 2 estratégias
   python trabalho_pai.py --treinar --redes resnet50 --regimes 3fases --estrategias central
   python trabalho_pai.py --ajustar       # tentativas de ajuste de hiperparâmetros
   ```
4. Saídas:
   - `pesos/<rede>_<regime>_<estrategia>.pt`: pesos da melhor época e limiares de decisão (não vão para o git);
   - `resultados/metricas.csv`: métricas do treino final no teste;
   - `resultados/ajuste.csv` e `ajuste_resumo.csv`: todas as tentativas do ajuste;
   - `resultados/loss_*.png`: curvas de convergência;
   - `resultados/gradcam_*.png`: exemplos de Grad-CAM de acertos e erros.

---

## 1. O que descobrimos sobre o dataset

- **172 pacientes**. Cada um tem 3 arquivos de fase em `dce/` e 1 máscara em `mask/`.
- Nome da fase: `{pid}_spy1_vis1_dce_aqc_{i}.nii.gz`. Em 141 pacientes o `i` é `0/1/2`, mas em 31 é `0/2/3`
  ou `0/2/5`. Como cada paciente tem exatamente 3 arquivos e a ordem do índice é a ordem temporal,
  **o programa usa os arquivos que existem, em ordem** (pré, pós precoce, pós tardio).
  Isso bate com as colunas `pre`, `post_early` e `post_late` do CSV em 171 pacientes. A exceção é o
  **ISPY1_1139**: o CSV diz `0/2/2` (repete a fase tardia no lugar da precoce), mas os arquivos são `0/1/2`.
- Shape de cada volume: **(N, 256, 256)**. **O eixo 0 é o dos cortes (direção z)** e cada corte já é o plano
  xy de 256×256. N vai de 10 a 72 (média 30).
- O cabeçalho NIfTI tem matriz afim identidade, então não traz orientação anatômica. Mostrar `volume[k]`
  direto dá a mesma orientação da Figura 1 do enunciado (mama apontando para a esquerda).
- O volume já vem **recortado em volta do tumor** ("MinCrop"). O corte do meio (`N // 2`) contém tumor nos
  172 pacientes. Cada paciente tem de 6 a 68 cortes com tumor (média 25,7; 4.420 no total).
  Os campos `n_z`, `mask_start` e `mask_end` do CSV se referem ao exame **original** e não servem aqui.
- **O tumor é pequeno dentro da imagem:** ocupa em média 2,0% da área do corte central (mediana 1,3%).
- Intensidades **sem escala fixa**: o máximo do volume vai de 53 a 7.255 (mediana 300) → precisa normalizar.
  Um volume, o pré-contraste do ISPY1_1187, tem uma região saturada em 65.535.
- Coluna `test` do CSV (artigo do dataset, Tabela 4): **0 = treino, 1 = teste, 2 = validação**.

| Partição | Pacientes | sem pCR | com pCR | Cortes com tumor |
|---|---|---|---|---|
| treino (`test=0`) | 104 | 78 | 26 | 2.551 |
| validação (`test=2`) | 33 | 22 | 11 | 904 |
| teste (`test=1`) | 35 | 23 | 12 | 965 |

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

**Etapa 5 — Avaliação:** métricas de sensibilidade, especificidade, precisão, acurácia e F1, mais a AUC.
O limiar de decisão é avaliado de duas formas: fixo em 0,5 e **escolhido na validação**.
Na estratégia "todos os cortes", junta os cortes por paciente (média ou máximo).

**Etapa 6 — Grad-CAM:** mapa de calor de onde a rede olhou, com o contorno do tumor, e uma medida de quanto
do mapa cai dentro do tumor.

**Etapa 7 — Experimentos:** 3 redes × 2 regimes × 2 estratégias = 12 treinos. Tudo vai para `resultados/`.

**Etapa 8 — Ajuste de hiperparâmetros:** roda uma lista de tentativas, cada uma com 3 sementes, e resume.

**Etapa 9 — Interface gráfica:** visualização com a máscara ao lado, aumento de dados, desbalanceamento,
treino e classificação com Grad-CAM.

---

## 3. Função por função

### Configurações (topo do arquivo)

| Nome | O que é |
|---|---|
| `PASTA_RAIZ`, `PASTA_BASE`, ... | caminhos montados a partir do próprio arquivo. Funcionam no Mac e no Windows sem editar |
| `MARCA_FASE`, `MARCA_MASCARA` | pedaços do nome dos arquivos (`_dce_aqc_` e `_mask`) usados para achar as fases e a máscara de um exame |
| `FASES` | nome das 3 fases, na ordem dos canais |
| `PARTICAO_TREINO/TESTE/VALIDACAO` | 0, 1, 2: significado da coluna `test` |
| `REGIMES` | `"1fase"` usa só o canal 1 (pós-contraste precoce); `"3fases"` usa os 3 |
| `ANGULOS`, `VARIACOES` | as 10 combinações (ângulo, espelhar) do aumento de dados |
| `HIPERPARAMETROS` | dicionário com taxa de aprendizado, decaimento de pesos, dropout, tamanho do lote e semente do treino final |
| `LIMIAR_PADRAO`, `EPOCAS`, `PACIENCIA` | limiar 0,5; máximo de épocas e paciência do early stopping por estratégia |
| `TENTATIVAS`, `SEMENTES_AJUSTE` | lista de tentativas do ajuste (o que cada uma muda) e as 3 sementes usadas |
| `dispositivo` | MPS (Mac), CUDA (NVIDIA) ou CPU, o que estiver disponível |

### `fixar_semente(semente)`
Fixa os sorteios do `random`, do `numpy` e do `torch`. Assim, rodar de novo dá o mesmo resultado, e
a comparação entre 1 fase e 3 fases é justa (mesma inicialização, mesma ordem dos lotes).

### Etapa 1 — Leitura

**`ler_volume(caminho)`**: abre o `.nii.gz` com `nibabel` e devolve o array 3D em `float32`
(metade da memória do `float64` e o tipo que o PyTorch usa).

**`normalizar(volume)`**: z-score: `(volume − média) / desvio`. Cada volume passa a ter média 0 e desvio 1.
O `+ 1e-8` evita divisão por zero. A normalização é **por volume** porque cada exame tem sua escala.

**`ler_fases(caminhos)`**: lê e normaliza os 3 arquivos de fase e devolve `(3, N, 256, 256)`:
fase × corte × altura × largura.

**`localizar_csv(pasta_dataset)`** e **`ler_tabela(pasta_dataset)`**: procuram a planilha em três lugares
(ao lado da pasta do dataset, dentro dela e ao lado do `.py`) e leem com `pandas`.

**`indice_da_fase(nome_arquivo)`**: devolve o número `N` do sufixo `_dce_aqc_N`.

**`caminhos_das_fases(pid, pasta_dataset)`**: lista os arquivos de fase do paciente em `dce/` e ordena pelo
índice. É aqui que fica a regra "usar os arquivos que existem, em ordem" da seção 1.

**`ler_mascara(pid, pasta_dataset)`**: lê a máscara do tumor (1 = tumor, 0 = fundo).

**`carregar_dados(estrategia)`**: percorre os 172 pacientes do CSV:
- `"central"`: pega 1 corte, `N // 2`;
- `"todos"`: pega todos os cortes `k` em que a máscara tem tumor.

Para cada corte escolhido, guarda a imagem `(3, 256, 256)` em `float16`, a máscara do corte, o rótulo `pCR`,
a partição e o `pid`. O `pid` serve para juntar depois as previsões de cada paciente, e a máscara serve para
avaliar o Grad-CAM. Devolve um dicionário de arrays: `imagens`, `mascaras`, `rotulos`, `particoes` e `pacientes`.

**`localizar_exame(caminho_arquivo)`**: usada pela interface. A partir do arquivo em que o usuário clicou,
descobre o nome do exame (o que vem antes de `_dce_aqc_` ou de `_mask`) e procura as 3 fases e a máscara na
própria pasta e nas pastas vizinhas `dce/` e `mask/`. Não depende do CSV, então funciona para um paciente novo.

**`verificar_desbalanceamento(tabela)`**: conta os pacientes sem pCR e com pCR em cada partição e no total.
No treino dá 78 para 26, ou seja, 3 para 1.

### Etapa 2 — Aumento de dados e Dataset

**`aplicar_variacao(imagem, angulo, espelhar)`**: espelha (inverte as colunas) e gira com
`scipy.ndimage.rotate` (`reshape=False` mantém 256×256, `mode="nearest"` preenche os cantos com o pixel mais
próximo, `order=1` é interpolação bilinear).
- aceita **vários canais** e gira as 3 fases juntas (`axes=(2, 1)`), para elas continuarem alinhadas;
- gera **uma** variação por chamada.

**`gerar_aumentos(corte)`**: gera as 10 variações de um corte 2D chamando `aplicar_variacao` para cada item de
`VARIACOES`. É o que a interface mostra no botão "Aumento de dados".

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
| 5 | FC 128 → 64 + ReLU + Dropout | 64 |
| 6 | FC 64 → 1 | 1 (logit) |

- ~106 mil parâmetros, **todos treinados**.
- `forward`: extrator → média global → classificador.
- `camada_gradcam`: a ReLU do último bloco convolucional.

**`repetir_para_3_canais(x)`**: se a entrada tem 1 canal, repete 3 vezes. As redes do ImageNet foram
treinadas com fotos RGB e a primeira camada espera 3 canais. Repetir mantém os pesos pré-treinados.

**`class ResNet50Congelada`**:
- carrega a ResNet50 com os pesos do ImageNet (`IMAGENET1K_V2`);
- congela todos os parâmetros (`requires_grad = False`);
- troca a `fc` (2048 → 1000) por `Dropout + Linear(2048 → 1)`. Só ela treina: 2.049 parâmetros de 23,5 milhões.
- `train(mode)`: sobrescrita para a base **ficar sempre em modo de avaliação**. Sem isso, o BatchNorm da base
  recalcularia as médias com os nossos lotes, mesmo congelado, e estragaria o que foi aprendido no ImageNet.
- `forward`: repete o canal, se preciso, e passa pela rede.
- `camada_gradcam`: `layer4`, o último estágio convolucional (2048 mapas de 8×8 para a entrada de 256).
- `pre_treinada=False`: cria a rede sem baixar os pesos do ImageNet. É usado ao recarregar um modelo já
  treinado, cujo arquivo já contém a rede inteira. Assim a apresentação não depende de internet.

**`class VGG16Congelada`**:
- carrega a VGG16 com os pesos do ImageNet e congela tudo;
- troca o `avgpool` por média global (512 valores) e o `classifier` (~120 milhões de parâmetros) por
  `FC 512 → 64 + ReLU + Dropout + FC 64 → 1` (~33 mil parâmetros). Treinar 120 milhões com 104 pacientes seria decorar;
- deixa as ReLU com `inplace=False`, o que é necessário para o Grad-CAM ler as ativações;
- `train(mode)`: mantém as convoluções (`features`) em modo de avaliação;
- `camada_gradcam`: `features[29]`, a ReLU da última convolução.

**`REDES`**: dicionário nome → classe (`"cnn"`, `"resnet50"`, `"vgg16"`), para escolher a rede pelo nome.

**`contar_parametros(modelo)`**: devolve (treináveis, total). Vai para o CSV de métricas e para o relatório.

**`caminho_dos_pesos(rede, regime, estrategia)`**: nome do arquivo de pesos de cada modelo.

**`salvar_modelo(modelo, caminho, limiares)`**: grava um dicionário com os pesos (`state_dict`) e os limiares
de decisão escolhidos na validação. Um arquivo só leva tudo o que a interface precisa.

**`carregar_modelo(rede, regime, estrategia)`**: recria a rede (sem baixar o ImageNet), carrega os pesos,
põe em modo de avaliação e devolve também os limiares. É o que a interface usa para classificar sem treinar.

### Etapa 4 — Treino

**`calcular_peso_positivo(rotulos_treino)`**: `nº sem pCR / nº com pCR` no **treino** = 78/26 = **3**.
É o tratamento do desbalanceamento: errar um paciente com pCR pesa 3× na loss.

**`executar_epoca(modelo, carregador, funcao_perda, otimizador=None)`**: uma passada por todos os lotes.
- Com otimizador (treino), em cada lote: forward → loss → `zero_grad` → `backward` → `step`.
- Sem otimizador (validação): só forward e loss, sem calcular gradiente (`set_grad_enabled(False)`).
- Devolve a loss média da época.

**`treinar(...)`**:
1. cria a loss `BCEWithLogitsLoss(pos_weight=3)` e o otimizador Adam, **só com os parâmetros treináveis**,
   com a taxa de aprendizado e o decaimento de pesos do dicionário `hiper`;
2. a cada época treina, valida e guarda as duas losses;
3. se a loss de validação melhorou, salva os pesos e zera o contador;
4. se ficar `PACIENCIA` épocas sem melhorar, para (**early stopping**);
5. no fim, recarrega os pesos da melhor época e devolve o histórico (losses, melhor época, tempo).

### Etapa 5 — Avaliação

**`prever(modelo, carregador)`**: modo de avaliação, sem gradiente. Aplica a sigmoide nos logits e devolve
as probabilidades de pCR.

**`agregar_por_paciente(probabilidades, rotulos, pacientes, modo)`**: estratégia "todos os cortes".
Agrupa as probabilidades pelo `pid` e tira a **média** ou o **máximo**. Devolve uma probabilidade e um rótulo por paciente.

**`calcular_metricas(rotulos, probabilidades, limiar)`**: probabilidade ≥ limiar → pCR. Monta a matriz de
confusão (VP, VN, FP, FN) e calcula:
- sensibilidade = VP/(VP+FN);
- especificidade = VN/(VN+FP);
- precisão = VP/(VP+FP);
- acurácia = (VP+VN)/total;
- F1 = 2·P·S/(P+S).

A função `dividir` devolve 0 quando o denominador é 0, por exemplo se a rede nunca prever pCR.

**`calcular_auc(rotulos, probabilidades)`**: área sob a curva ROC. Mede se a rede dá probabilidades maiores
para quem tem pCR, **sem depender do limiar**. 0,5 é o acaso e 1,0 é a separação perfeita.

**`escolher_limiar(rotulos, probabilidades)`**: testa como limiar os pontos médios entre probabilidades
vizinhas e fica com o que maximiza o **índice de Youden** (sensibilidade + especificidade − 1).
- É chamada com as previsões da **validação**; o limiar é depois aplicado ao teste.
- Por que não maximizar o F1: com 1/3 de positivos, dizer "pCR" para todo mundo já dá F1 = 0,50.
  O Youden dá o mesmo peso às duas classes e não premia essa resposta.
- Em empate, vence o limiar mais perto de 0,5.

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

**`desenhar_gradcam(eixo, corte, mapa, mascara)`**: desenha o corte em cinza, o mapa de calor por cima e o
contorno do tumor em branco. É usada tanto nas figuras salvas quanto na interface.

**`salvar_figura_gradcam(...)`**: três painéis: a imagem (pós-contraste precoce), o Grad-CAM sobreposto e a máscara
do tumor. Assim dá para ver se a rede olhou para o tumor (pergunta do relatório).

**`medir_gradcam_no_tumor(modelo, imagens, mascaras, canais)`**: responde com número à pergunta "o Grad-CAM
aponta para o tumor?". Para cada imagem do teste:
- **pico no tumor**: o ponto mais quente do mapa está dentro da máscara? (sim/não);
- **calor no tumor**: que fração do calor total do mapa está dentro da máscara;
- **área do tumor**: que fração da imagem o tumor ocupa. É a referência: um mapa espalhado ao acaso teria
  "calor no tumor" igual à "área do tumor".

### Etapa 7 — Experimentos

**`separar(dados, particao)`**: filtra todos os arrays de uma partição.

**`executar_experimento(nome_rede, regime, estrategia, dados, hiper, tentativa)`**: um experimento completo:
1. junta os hiperparâmetros recebidos com os de `HIPERPARAMETROS`, fixa a semente e separa treino, validação e teste;
2. cria os `DataLoader`. O de treino é embaralhado e com sorteio de variação. O de validação usa as 10 variações
   na estratégia central e a imagem original em "todos" (validar 10× todos os cortes multiplicaria o tempo por 10);
3. cria a rede com 1 ou 3 canais e treina;
4. prevê a validação e o teste **sem aumento**. Na estratégia central, cada imagem é um paciente; em "todos",
   agrega por média e por máximo;
5. escolhe o limiar na validação e calcula as métricas do teste com o limiar 0,5 e com o limiar escolhido;
6. no treino final (`tentativa` vazia), mede o Grad-CAM no tumor, regrava os pesos com os limiares e salva a curva de loss.
   Nas tentativas de ajuste, os pesos vão para um arquivo temporário;
7. devolve as linhas de métricas, o modelo, o histórico e os limiares.

**`salvar_metricas(linhas, caminho)`**: grava as linhas no CSV. Se o mesmo experimento (mesma tentativa, rede,
regime, estratégia e semente) já estiver no arquivo, as linhas antigas são substituídas.

**`gerar_exemplos_gradcam(modelo, nome, regime, dados, limiar)`**: percorre o teste e salva uma figura de
Grad-CAM para o primeiro exemplo de cada tipo: VP e VN (acertos), FP e FN (erros).

**`treinar_redes(redes, regimes, estrategias, hiper)`**: o treino final. Para cada estratégia, carrega os
dados uma vez e roda todas as combinações de rede × regime.

### Etapa 8 — Ajuste de hiperparâmetros

**`ajustar_hiperparametros(redes, regimes)`**: roda cada item de `TENTATIVAS` com as 3 sementes de
`SEMENTES_AJUSTE`, nas 3 redes e nos 2 regimes, sempre na estratégia do corte central (a mais rápida).
Tudo é registrado em `resultados/ajuste.csv`.

**`resumir_ajuste()`**: lê o CSV e calcula, para cada tentativa, rede e regime, a média e o desvio padrão
entre as sementes. O desvio padrão mostra o tamanho do ruído: uma diferença entre tentativas menor do que
ele não quer dizer nada.

**`salvar_comparacao_de_convergencia(...)`**: uma figura com a loss de validação (linha cheia) e de treino
(tracejada) de cada tentativa, para comparar a velocidade de convergência e o overfitting.

**Regra do ajuste:** a escolha olha só para a **validação**. O teste fica guardado para o treino final.

### Etapa 9 — Interface gráfica

**`class VisualizadorNii`** (área da direita):
- `carregar_arquivo(caminho)`: lê o volume, chama `localizar_exame` para achar a máscara e começa no corte do meio;
- `montar_visualizacao()`: cria a figura com três painéis (imagem, segmentação, Grad-CAM) e o controle deslizante.
  O brilho usa a mesma janela em todos os cortes (do mínimo ao percentil 99,5 do volume);
- `atualizar_corte(valor)`: troca o corte da imagem **e da máscara** quando o controle se move, e limpa o
  Grad-CAM, que era de outro corte;
- `mostrar_gradcam(mapa, titulo)`: desenha o mapa no terceiro painel.

**`class EscritorDeFila`**: recebe o texto dos `print()` da thread de treino e põe em uma fila, para a
janela de treino mostrar sem travar a interface.

**`class InterfaceMRI`** (janela e painel da esquerda):
- `selecionar_pasta` / `listar_arquivos`: lista os `.nii` da pasta. Se for a pasta do dataset, lista o que está em `dce/`;
- `classificar`: carrega o modelo escolhido nos menus (`obter_modelo`), lê as 3 fases do exame (`obter_fases`),
  gera o Grad-CAM do corte mostrado e escreve a probabilidade, o limiar, a classe prevista e a classe real.
  Na estratégia "todos os cortes", a probabilidade do paciente é a média das previsões de todos os cortes com tumor;
- `mostrar_aumento_de_dados`: janela com as 10 variações do corte;
- `mostrar_desbalanceamento`: janela com a contagem por partição e o peso usado na loss;
- `treinar_rede_selecionada` / `atualizar_log`: treina em uma thread separada e mostra o andamento.

### `main()`
Lê as opções da linha de comando. Sem opções abre a interface; `--treinar` roda o treino final e
`--ajustar` roda o ajuste. Os hiperparâmetros podem ser trocados por `--taxa-aprendizado`,
`--decaimento-pesos`, `--dropout`, `--tamanho-lote` e `--semente`.

---

## 4. Hiperparâmetros e por quê

| Hiperparâmetro | Valor | Por quê |
|---|---|---|
| Fase do regime "1 fase" | pós-contraste precoce | é a fase em que o tumor mais realça (pico do contraste) |
| Normalização | z-score por volume | cada exame tem escala própria; média 0 e desvio 1 também é a escala que as redes do ImageNet esperam |
| Tamanho da imagem | 256×256 (sem redimensionar) | ResNet e VGG terminam em pooling adaptativo, então aceitam 256 |
| Tamanho do lote | 16 | o treino tem 104 imagens (≈ 7 lotes por época); no ajuste, lote 32 (T7) não mudou a validação |
| Otimizador | Adam | ajusta o passo de cada peso sozinho; converge rápido com pouco ajuste |
| Taxa de aprendizado | 1e-3 | no ajuste, foi a de menor loss de validação entre 1e-3, 3e-4, 1e-4 e 1e-5 (T1 a T4). Com 1e-5 a rede nem sai do lugar (underfitting) |
| Weight decay | **1e-2** | era 1e-4 na primeira rodada. No ajuste (T5), 1e-2 deu a menor loss de validação; o ganho veio da CNN própria |
| Dropout | 0,5 | valor clássico; no ajuste, 0,7 (T6) não trouxe ganho |
| Loss | BCE com `pos_weight = 3` | problema binário + desbalanceamento 78:26 no treino |
| Limiar | escolhido na validação (índice de Youden) | com 0,5 fixo, o peso 3 da classe pCR faz as redes dizerem "pCR" demais. As métricas com 0,5 também ficam no CSV |
| Épocas máx. / paciência | central: 40 / 8; todos: 15 / 4 | "todos" tem ~25× mais imagens por época, então converge em menos épocas |
| Congelamento | ResNet50: só a FC treina; VGG16: só a cabeça | o enunciado pede isso na ResNet50; com 104 pacientes, treinar milhões de pesos decoraria o treino |
| Cabeça da VGG16 | média global + 512→64→1 | a cabeça original (~120 milhões) seria grande demais |
| CNN própria | 4 conv + 2 FC, filtros 16→128 | ≥ 5 camadas com parâmetros (exigência), pequena (~106 mil) pelo pouco dado; filtros 3×3 como na VGG; média global no lugar de uma FC grande |
| Semente | 42 (no ajuste: 42, 43 e 44) | reprodutibilidade; as 3 sementes do ajuste medem o ruído |

---

## 5. Resultados

Tudo foi rodado em um Mac com M4 Pro (MPS). Os números completos estão em `resultados/metricas.csv`
(treino final) e `resultados/ajuste.csv` (ajuste). A primeira rodada, de 30/10, está em `resultados/rodada1/`.

### 5.1 Ajuste de hiperparâmetros (05/11)

7 tentativas × 3 redes × 2 regimes × 3 sementes = **126 treinos**, todos no corte central.
A AUC é a média ± desvio padrão de 6 treinos; a loss de validação e a melhor época são a média dos 18.

| Tent. | Taxa | Weight decay | Dropout | Lote | AUC val. CNN | AUC val. ResNet50 | AUC val. VGG16 | Loss val. | Melhor época |
|---|---|---|---|---|---|---|---|---|---|
| T1 | 1e-3 | 1e-4 | 0,5 | 16 | 0,42 ± 0,18 | 0,47 ± 0,05 | 0,55 ± 0,02 | 1,142 | 6,1 |
| T2 | 3e-4 | 1e-4 | 0,5 | 16 | 0,44 ± 0,10 | 0,46 ± 0,08 | 0,51 ± 0,06 | 1,153 | 6,5 |
| T3 | 1e-4 | 1e-4 | 0,5 | 16 | 0,40 ± 0,13 | 0,45 ± 0,09 | 0,54 ± 0,10 | 1,155 | 6,3 |
| T4 | 1e-5 | 1e-4 | 0,5 | 16 | 0,43 ± 0,12 | 0,46 ± 0,10 | 0,55 ± 0,09 | 1,155 | 12,2 |
| **T5** | 1e-3 | **1e-2** | 0,5 | 16 | 0,59 ± 0,20 | 0,47 ± 0,05 | 0,54 ± 0,02 | **1,136** | 7,9 |
| T6 | 1e-3 | 1e-4 | **0,7** | 16 | 0,50 ± 0,19 | 0,46 ± 0,08 | 0,56 ± 0,06 | 1,146 | 7,6 |
| T7 | 1e-3 | 1e-4 | 0,5 | **32** | 0,51 ± 0,14 | 0,47 ± 0,05 | 0,54 ± 0,06 | 1,143 | 6,3 |

**Leitura honesta:**
- **Nenhuma tentativa tira as redes do acaso na validação.** Uma rede que responde 0,5 para todo mundo tem
  loss de validação de 1,155; todas as tentativas ficaram a menos de 0,02 disso.
- **O ruído é maior que as diferenças.** O desvio padrão da AUC entre sementes chega a 0,20.
- **Taxa menor não ajuda.** Com 1e-5 a loss de treino termina em 1,04–1,05, onde começou (underfitting).
  Com 1e-3 a loss de treino cai, mas a de validação sobe depois de poucas épocas (overfitting), e o early stopping para.
- **T5 foi a escolhida** (menor loss de validação e maior AUC média, 0,53). O ganho é da CNN própria;
  nas redes congeladas o weight decay quase não muda nada, porque só a cabeça treina.
- Conclusão do ajuste: o limite do desempenho **não está nesses hiperparâmetros**.

A figura `resultados/ajuste_convergencia.png` mostra as curvas das 7 tentativas.

### 5.2 Treino final (05/11, configuração T5, semente 42)

Conjunto de **teste** (35 pacientes: 23 sem pCR, 12 com pCR), com o **limiar escolhido na validação**.

| Rede | Regime | Cortes | Limiar | Sens. | Espec. | Prec. | Acur. | F1 | AUC | Tempo (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| CNN | 1 fase | central | 0,55 | 0,33 | 0,48 | 0,25 | 0,43 | 0,29 | 0,44 | 19 |
| CNN | 3 fases | central | 0,50 | 0,50 | 0,52 | 0,35 | 0,51 | 0,41 | 0,56 | 18 |
| ResNet50 | 1 fase | central | 0,51 | 0,67 | 0,35 | 0,35 | 0,46 | 0,46 | 0,53 | 19 |
| ResNet50 | 3 fases | central | 0,54 | 0,42 | 0,87 | 0,62 | **0,71** | 0,50 | **0,67** | 28 |
| VGG16 | 1 fase | central | 0,51 | 0,42 | 0,65 | 0,38 | 0,57 | 0,40 | 0,49 | 35 |
| VGG16 | 3 fases | central | 0,52 | 0,08 | 0,78 | 0,17 | 0,54 | 0,11 | 0,57 | 44 |
| CNN | 1 fase | média | 0,51 | 0,75 | 0,09 | 0,30 | 0,31 | 0,43 | 0,52 | 57 |
| CNN | 3 fases | média | 0,46 | 0,83 | 0,30 | 0,38 | 0,49 | 0,53 | 0,62 | 69 |
| ResNet50 | 1 fase | média | 0,66 | 0,25 | 0,78 | 0,38 | 0,60 | 0,30 | 0,61 | 91 |
| ResNet50 | 3 fases | média | 0,56 | 0,17 | 0,91 | 0,50 | 0,66 | 0,25 | 0,66 | 90 |
| VGG16 | 1 fase | média | 0,49 | 0,33 | 0,78 | 0,44 | 0,63 | 0,38 | 0,63 | 114 |
| VGG16 | 3 fases | média | 0,49 | 0,42 | 0,65 | 0,38 | 0,57 | 0,40 | 0,63 | 137 |
| CNN | 1 fase | máximo | 0,52 | 0,25 | 0,52 | 0,21 | 0,43 | 0,23 | 0,33 | 57 |
| CNN | 3 fases | máximo | 0,52 | 0,83 | 0,26 | 0,37 | 0,46 | 0,51 | **0,67** | 69 |
| ResNet50 | 1 fase | máximo | 0,80 | 0,17 | 0,74 | 0,25 | 0,54 | 0,20 | 0,55 | 91 |
| ResNet50 | 3 fases | máximo | 0,61 | 0,42 | 0,74 | 0,45 | 0,63 | 0,43 | 0,63 | 90 |
| VGG16 | 1 fase | máximo | 0,50 | 0,83 | 0,48 | 0,45 | 0,60 | **0,59** | 0,61 | 114 |
| VGG16 | 3 fases | máximo | 0,54 | 0,25 | 0,78 | 0,38 | 0,60 | 0,30 | 0,64 | 137 |

**Leitura honesta:**
- **O desempenho continua perto do acaso.** A AUC vai de 0,33 a 0,67. Responder "sem pCR" para todos dá
  acurácia de 0,66 no teste; só a ResNet50 de 3 fases no corte central passa disso (0,71).
- **Melhor classificador:** ResNet50, 3 fases, corte central. Mas com 35 pacientes o intervalo de confiança
  da AUC (0,67) vai de mais ou menos 0,47 a 0,87, ou seja, inclui o acaso.
- **As saídas são pouco separadas:** no corte central, 80% das probabilidades das redes pré-treinadas
  ficam entre 0,48 e 0,56.
- **O limiar faz diferença.** Com 0,5 fixo, a acurácia média dos 18 classificadores é 0,48 e metade tem
  especificidade abaixo de 0,35. Com o limiar da validação a acurácia média vai a 0,54.
  O F1 médio cai (0,48 → 0,37), porque o F1 premia dizer "pCR" para muita gente.
- **Tempo de treino:** 18 a 44 s por modelo no corte central, 57 a 137 s com todos os cortes; 12 minutos no total.
- **Curvas:** a loss de treino cai e a de validação sobe desde as primeiras épocas (overfitting).
  A CNN de 1 fase com todos os cortes não aprendeu nada (loss de treino parada em 1,05): para ela o weight decay
  de 1e-2 ficou forte demais.

### 5.3 Grad-CAM e o tumor

"Pico" = fração das imagens de teste em que o ponto mais quente do mapa cai dentro do tumor.
"Calor" = fração do calor do mapa dentro do tumor. "Área" = o que o calor valeria se o mapa fosse ao acaso.

| Rede | Treino | Pico 1 fase | Pico 3 fases | Calor 1 fase | Calor 3 fases | Área do tumor |
|---|---|---|---|---|---|---|
| CNN | corte central | 0% | 0% | 0,5% | 0,1% | 1,6% |
| ResNet50 | corte central | 9% | 3% | 2,8% | 2,5% | 1,6% |
| VGG16 | corte central | 9% | 23% | 3,3% | 4,7% | 1,6% |
| CNN | todos os cortes | 47% | 15% | 7,1% | 6,1% | 1,2% |
| ResNet50 | todos os cortes | 1% | 2% | 1,2% | 2,1% | 1,2% |
| VGG16 | todos os cortes | 2% | 5% | 1,1% | 2,5% | 1,2% |

Em geral o Grad-CAM **não** aponta para o tumor. A exceção é a CNN treinada com todos os cortes, que olha
para o tumor 5 a 6 vezes mais do que o acaso, mas nem por isso classifica melhor.

### 5.4 Respostas curtas às perguntas do relatório

1. **3 fases é melhor que 1 fase?** No treino final, a AUC de teste com 3 fases foi maior nos 9 pares
   (média 0,63 contra 0,52). Mas a diferença (0,10) é do tamanho do erro padrão da AUC com 35 pacientes,
   e nos 63 pares do ajuste ela cai para 0,02 no teste e −0,02 na validação. **Há tendência, não há prova.**
2. **Melhor rede?** ResNet50 e VGG16 empatam (acurácia média 0,60 e 0,59) e ficam acima da CNN própria (0,44).
3. **Erros mais frequentes?** Com limiar 0,5, falsos positivos (65% dos sem pCR). Com o limiar da validação,
   a taxa de falsos negativos passa a ser a maior (56% dos com pCR, contra 41% de falsos positivos).
4. **Grad-CAM e tumor?** Pouca relação (tabela acima).
5. **Limitações?** Poucos pacientes (104/33/35); o tumor ocupa ~2% da imagem e a rede não sabe onde ele está;
   bases congeladas treinadas em fotos; cada corte tratado isoladamente; uma semente só no treino final.

---

## 6. De onde veio cada parte

Na junção em um arquivo, o código de cada um foi mantido o mais parecido possível com o original.

| Código original | Onde está agora | O que mudou |
|---|---|---|
| `trabalho/redesNeurais/neuralNetwork.py` (Bernardo) | Etapas 1 a 7 | hiperparâmetros viraram um dicionário, para o ajuste; as fases passaram a vir dos arquivos em ordem (caso do ISPY1_1139); `carregar_dados` também guarda as máscaras; entraram a AUC, o limiar escolhido na validação e a medida do Grad-CAM no tumor; o arquivo de pesos passou a guardar os limiares |
| `trabalho/aumentoDeDados.py` e `Rotação.py` (Kauan) | `gerar_aumentos` | agora chama `aplicar_variacao`, para existir um código só de rotação, igual ao usado no treino. A interpolação passou de cúbica (padrão do SciPy) para bilinear |
| `teste_desbalanceamento.py` (Kauan) | `verificar_desbalanceamento` | conta por partição e devolve os números em vez de imprimir; a interface mostra |
| `visualizador.py` (Kauan e Diego) | `VisualizadorNii` | ganhou a máscara ao lado e o painel do Grad-CAM. **O menu Axial/Coronal/Sagital saiu:** o enunciado pede o plano xy navegando em z, e nos arquivos do dataset esse plano é `volume[k]` (o que o menu chamava de "Sagital"). O padrão antigo ("Axial") mostrava uma faixa de N×256 |
| `trabalho/interface.py` (Diego) | `InterfaceMRI` | ganhou os menus de rede, regime e estratégia e os botões de classificar, aumento de dados, desbalanceamento e treino |
| `extrairCorteCentral.py` e `extrairTodosOsCortes.py` (Diego) | dentro de `carregar_dados` | o corte central é `N // 2` no eixo 0 e "todos" são os cortes com tumor |
| `redesNeurais/extrairDatasets.py` e `test.py` (Diego) | removidos | tinham o caminho fixo do computador do Diego e a função deles ficou em `carregar_dados` |

Os arquivos originais continuam nas branches `Bernardo`, `Diego` e `Kauan`.

---

## 7. O que ainda falta

- **Preencher os dados do grupo:** nomes completos, matrículas, campus e número do grupo no topo do
  `trabalho_pai.py`, no `relatorio/relatorio.tex` e na `declaracao_uso_iag.md`.
- **Declaração de uso de IA:** cada um preenche a sua parte.
- **Repositório privado:** o enunciado exige; só o dono (Kauan) consegue mudar.
- **Conferir a exibição no MRIcro**, como pede o enunciado.
- **Pesos para a apresentação:** a pasta `pesos/` não vai para o git. Quem for apresentar precisa rodar
  `python trabalho_pai.py --treinar` no próprio computador, ou copiar a pasta.
- **Montar o ZIP:** `Gnn.zip` com o `.py`, o relatório (`.tex`, `.sty`, figuras e PDF) e a declaração, até 10 MB,
  sem dataset e sem pesos.
- **Opcional do enunciado:** máscara como canal extra ou fundo zerado.
