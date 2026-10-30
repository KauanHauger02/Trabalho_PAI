# Redes neurais para classificar a resposta patológica completa (pCR) a partir de DCE-MRI.
#
# Três redes (CNN própria, ResNet50 e VGG16), cada uma treinada em dois regimes:
#   - "1fase":  só o pós-contraste precoce (1 canal)
#   - "3fases": pré-contraste, pós-contraste precoce e pós-contraste tardio (3 canais)
# e com duas estratégias para passar do volume 3D para imagens 2D:
#   - "central": só o corte do meio de cada volume
#   - "todos":   todos os cortes com tumor; a previsão do paciente é a média ou o máximo dos cortes
#
# Como rodar (a partir da raiz do repositório):
#   .venv/bin/python trabalho/redesNeurais/neuralNetwork.py
#   .venv/bin/python trabalho/redesNeurais/neuralNetwork.py --redes resnet50 --estrategias central
#
# Explicação detalhada de cada função: ENTENDIMENTO.md, na raiz do repositório.

import argparse
import csv
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # gera os gráficos direto em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

PASTA_RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PASTA_BASE = os.path.join(PASTA_RAIZ, "Base de dados")
PASTA_DATASET = os.path.join(PASTA_BASE, "BreastDCEDL_ISPY1_min_crop")
CAMINHO_CSV = os.path.join(PASTA_BASE, "BreastDCEDL_SPY1_min_crop_metadata.csv")
PASTA_PESOS = os.path.join(PASTA_RAIZ, "pesos")          # não vai para o git (arquivos grandes)
PASTA_RESULTADOS = os.path.join(PASTA_RAIZ, "resultados")

# Colunas do CSV que dizem qual arquivo é cada fase (o sufixo _aqc_N do nome)
FASES = ["pre", "post_early", "post_late"]

# Coluna "test" do CSV (artigo do BreastDCEDL, Tabela 4): 0 = treino, 1 = teste, 2 = validação
PARTICAO_TREINO, PARTICAO_TESTE, PARTICAO_VALIDACAO = 0, 1, 2

# Canais usados em cada regime (posições dentro de FASES)
REGIMES = {
    "1fase": [1],         # pós-contraste precoce: fase em que o tumor mais realça
    "3fases": [0, 1, 2],  # as três fases como três canais
}

# Aumento de dados exigido no enunciado: 5 ângulos x (original, espelhada) = 10 variações
ANGULOS = [-20, -10, 0, 10, 20]
VARIACOES = [(angulo, espelhar) for angulo in ANGULOS for espelhar in (False, True)]

# Hiperparâmetros (justificativas no ENTENDIMENTO.md)
SEMENTE = 42
TAMANHO_LOTE = 16
TAXA_APRENDIZADO = 1e-3
DECAIMENTO_PESOS = 1e-4
LIMIAR = 0.5
EPOCAS = {"central": 40, "todos": 15}     # máximo de épocas
PACIENCIA = {"central": 8, "todos": 4}    # early stopping: épocas sem melhorar a loss de validação

if torch.backends.mps.is_available():
    dispositivo = torch.device("mps")     # GPU dos Macs com chip Apple
elif torch.cuda.is_available():
    dispositivo = torch.device("cuda")    # GPU NVIDIA
else:
    dispositivo = torch.device("cpu")


def fixar_semente(semente=SEMENTE):
    """Deixa os sorteios reproduzíveis, para comparar experimentos de forma justa."""
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)


# ---------------------------------------------------------------------------
# Etapa 1 - Leitura dos dados
# ---------------------------------------------------------------------------

def ler_volume(caminho):
    """Lê um arquivo .nii/.nii.gz e devolve o volume (cortes, 256, 256) em float32."""
    return nib.load(caminho).get_fdata().astype(np.float32)


def normalizar(volume):
    """Normalização z-score por volume: média 0 e desvio padrão 1.

    Cada exame tem uma escala de intensidade diferente (em um paciente o máximo é 110,
    em outro passa de 2000). Sem normalizar, a rede aprenderia a escala do aparelho.
    """
    return (volume - volume.mean()) / (volume.std() + 1e-8)


def ler_fases_do_paciente(linha):
    """Devolve as 3 fases normalizadas de um paciente, empilhadas: (3, cortes, 256, 256)."""
    fases = []
    for coluna in FASES:
        indice = int(linha[coluna])
        caminho = os.path.join(PASTA_DATASET, "dce", f"{linha['pid']}_spy1_vis1_dce_aqc_{indice}.nii.gz")
        fases.append(normalizar(ler_volume(caminho)))
    return np.stack(fases)


def ler_mascara(pid):
    """Devolve a máscara do tumor (cortes, 256, 256), com 1 = tumor e 0 = fundo."""
    return ler_volume(os.path.join(PASTA_DATASET, "mask", f"{pid}_spy1_vis1_mask.nii.gz"))


def carregar_dados(estrategia):
    """Lê todos os pacientes e devolve as imagens 2D com rótulo, partição e paciente.

    estrategia = "central": 1 imagem por paciente (o corte do meio do volume).
    estrategia = "todos":   1 imagem por corte que contém tumor.

    Retorna um dicionário com:
        imagens   (N, 3, 256, 256)  float16 (metade da memória; vira float32 na hora de usar)
        rotulos   (N,)              0 = sem pCR, 1 = com pCR
        particoes (N,)              0 = treino, 1 = teste, 2 = validação
        pacientes (N,)              pid do paciente de cada imagem
    """
    tabela = pd.read_csv(CAMINHO_CSV)
    imagens, rotulos, particoes, pacientes = [], [], [], []

    for numero, linha in tabela.iterrows():
        fases = ler_fases_do_paciente(linha)

        if estrategia == "central":
            cortes = [fases.shape[1] // 2]
        else:
            mascara = ler_mascara(linha["pid"])
            cortes = [k for k in range(mascara.shape[0]) if mascara[k].sum() > 0]

        for k in cortes:
            imagens.append(fases[:, k].astype(np.float16))
            rotulos.append(float(linha["pCR"]))
            particoes.append(int(linha["test"]))
            pacientes.append(linha["pid"])

        if (numero + 1) % 43 == 0:
            print(f"  lidos {numero + 1}/{len(tabela)} pacientes")

    return {
        "imagens": np.stack(imagens),
        "rotulos": np.array(rotulos, dtype=np.float32),
        "particoes": np.array(particoes),
        "pacientes": np.array(pacientes),
    }


# ---------------------------------------------------------------------------
# Etapa 2 - Aumento de dados e Dataset do PyTorch
# ---------------------------------------------------------------------------

def aplicar_variacao(imagem, angulo, espelhar):
    """Aplica uma das 10 variações a uma imagem (canais, 256, 256).

    Todos os canais (fases) recebem exatamente a mesma transformação, para continuarem alinhados.
    Mesmos parâmetros de rotação do aumentoDeDados.py do grupo: reshape=False mantém 256x256
    e mode="nearest" preenche os cantos com o pixel mais próximo.
    """
    if espelhar:
        imagem = imagem[:, :, ::-1]
    if angulo != 0:
        imagem = rotate(imagem, angle=angulo, axes=(2, 1), reshape=False, mode="nearest", order=1)
    return np.ascontiguousarray(imagem)


class DatasetPCR(Dataset):
    """Entrega pares (imagem, rótulo) para o DataLoader.

    modo_aumento:
        "sorteio" -> treino: a cada vez sorteia 1 das 10 variações (a rede vê todas ao longo das épocas
                     sem precisar guardar 10 cópias na memória)
        "todas"   -> validação: cada imagem aparece nas 10 variações, sempre as mesmas
        "nenhum"  -> teste: imagem original, sem aumento
    """

    def __init__(self, imagens, rotulos, canais, modo_aumento):
        self.imagens = imagens
        self.rotulos = rotulos
        self.canais = canais
        self.modo_aumento = modo_aumento

    def __len__(self):
        if self.modo_aumento == "todas":
            return len(self.imagens) * len(VARIACOES)
        return len(self.imagens)

    def __getitem__(self, posicao):
        if self.modo_aumento == "todas":
            indice, numero_variacao = divmod(posicao, len(VARIACOES))
            angulo, espelhar = VARIACOES[numero_variacao]
        elif self.modo_aumento == "sorteio":
            indice = posicao
            angulo, espelhar = random.choice(VARIACOES)
        else:
            indice, angulo, espelhar = posicao, 0, False

        imagem = self.imagens[indice][self.canais].astype(np.float32)
        imagem = aplicar_variacao(imagem, angulo, espelhar)
        return torch.from_numpy(imagem), torch.tensor(self.rotulos[indice])


# ---------------------------------------------------------------------------
# Etapa 3 - As três redes
# ---------------------------------------------------------------------------

def bloco_convolucional(entrada, saida):
    """Conv 3x3 + BatchNorm + ReLU + MaxPool 2x2: extrai padrões e reduz a imagem pela metade."""
    return nn.Sequential(
        nn.Conv2d(entrada, saida, kernel_size=3, padding=1),
        nn.BatchNorm2d(saida),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class CNNPropria(nn.Module):
    """CNN própria com 6 camadas com parâmetros (4 convolucionais + 2 totalmente conectadas).

    Entrada (C, 256, 256) -> 16 -> 32 -> 64 -> 128 filtros (a imagem cai de 256 para 16)
    -> média global de cada mapa (vetor de 128) -> FC 64 -> FC 1 (logit de pCR).
    """

    def __init__(self, n_canais):
        super().__init__()
        self.extrator = nn.Sequential(
            bloco_convolucional(n_canais, 16),
            bloco_convolucional(16, 32),
            bloco_convolucional(32, 64),
            bloco_convolucional(64, 128),
        )
        self.media_global = nn.AdaptiveAvgPool2d(1)
        self.classificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classificador(self.media_global(self.extrator(x)))

    def camada_gradcam(self):
        return self.extrator[-1][2]  # ReLU do último bloco convolucional


def repetir_para_3_canais(x):
    """As redes do ImageNet esperam 3 canais (RGB). No regime de 1 fase, repete o canal 3 vezes."""
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


class ResNet50Congelada(nn.Module):
    """ResNet50 pré-treinada no ImageNet: a base fica congelada e só a parte FC é treinada.

    A FC original (2048 -> 1000 classes do ImageNet) é trocada por Dropout + FC (2048 -> 1).
    """

    def __init__(self, n_canais):
        super().__init__()
        self.rede = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for parametro in self.rede.parameters():
            parametro.requires_grad = False
        self.rede.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 1))  # camada nova: treinável

    def train(self, mode=True):
        # A base congelada fica sempre em modo de avaliação, para o BatchNorm dela
        # não mudar as estatísticas aprendidas no ImageNet. Só a FC nova alterna treino/avaliação.
        super().train(mode)
        self.rede.eval()
        self.rede.fc.train(mode)
        return self

    def forward(self, x):
        return self.rede(repetir_para_3_canais(x))

    def camada_gradcam(self):
        return self.rede.layer4  # último estágio convolucional (2048 mapas 8x8)


class VGG16Congelada(nn.Module):
    """VGG16 pré-treinada no ImageNet: as 13 convoluções ficam congeladas.

    A cabeça original tem ~120 milhões de parâmetros (decoraria 104 pacientes). Ela é trocada por
    média global (512 valores) + FC 512 -> 64 + Dropout + FC 64 -> 1, com ~33 mil parâmetros.
    """

    def __init__(self, n_canais):
        super().__init__()
        self.rede = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for parametro in self.rede.parameters():
            parametro.requires_grad = False
        for camada in self.rede.features:
            if isinstance(camada, nn.ReLU):
                camada.inplace = False  # necessário para o Grad-CAM ler as ativações sem conflito
        self.rede.avgpool = nn.AdaptiveAvgPool2d(1)
        self.rede.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.rede.features.eval()
        return self

    def forward(self, x):
        return self.rede(repetir_para_3_canais(x))

    def camada_gradcam(self):
        return self.rede.features[29]  # ReLU da última convolução (512 mapas 16x16)


REDES = {
    "cnn": CNNPropria,
    "resnet50": ResNet50Congelada,
    "vgg16": VGG16Congelada,
}


def contar_parametros(modelo):
    """Devolve (parâmetros treináveis, parâmetros totais)."""
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    return treinaveis, total


# ---------------------------------------------------------------------------
# Etapa 4 - Treino
# ---------------------------------------------------------------------------

def calcular_peso_positivo(rotulos_treino):
    """Peso da classe pCR na loss = (nº sem pCR) / (nº com pCR) no treino.

    No treino são 78 sem pCR e 26 com pCR, então errar um paciente com pCR custa 3x mais.
    É o tratamento do desbalanceamento pedido no enunciado.
    """
    positivos = rotulos_treino.sum()
    negativos = len(rotulos_treino) - positivos
    return torch.tensor([negativos / positivos], dtype=torch.float32)


def executar_epoca(modelo, carregador, funcao_perda, otimizador=None):
    """Passa uma vez por todos os lotes. Com otimizador, treina; sem otimizador, só avalia.

    Retorna a loss média da época.
    """
    treinando = otimizador is not None
    modelo.train(treinando)
    soma_perda, quantidade = 0.0, 0

    with torch.set_grad_enabled(treinando):
        for imagens, rotulos in carregador:
            imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)
            logits = modelo(imagens).squeeze(1)       # (lote, 1) -> (lote,)
            perda = funcao_perda(logits, rotulos)

            if treinando:
                otimizador.zero_grad()   # zera os gradientes do lote anterior
                perda.backward()         # backpropagation: calcula os gradientes
                otimizador.step()        # atualiza os pesos

            soma_perda += perda.item() * len(rotulos)
            quantidade += len(rotulos)

    return soma_perda / quantidade


def treinar(modelo, carregador_treino, carregador_validacao, peso_positivo, max_epocas, paciencia, caminho_pesos):
    """Treina com early stopping e salva os pesos da época com menor loss de validação.

    Retorna o histórico: loss de treino e de validação por época, melhor época e tempo em segundos.
    """
    funcao_perda = nn.BCEWithLogitsLoss(pos_weight=peso_positivo.to(dispositivo))
    parametros_treinaveis = [p for p in modelo.parameters() if p.requires_grad]
    otimizador = torch.optim.Adam(parametros_treinaveis, lr=TAXA_APRENDIZADO, weight_decay=DECAIMENTO_PESOS)

    historico = {"perda_treino": [], "perda_validacao": [], "melhor_epoca": 0}
    melhor_perda = float("inf")
    epocas_sem_melhorar = 0
    inicio = time.time()

    for epoca in range(1, max_epocas + 1):
        perda_treino = executar_epoca(modelo, carregador_treino, funcao_perda, otimizador)
        perda_validacao = executar_epoca(modelo, carregador_validacao, funcao_perda)
        historico["perda_treino"].append(perda_treino)
        historico["perda_validacao"].append(perda_validacao)
        print(f"    época {epoca:2d}: loss treino {perda_treino:.4f} | loss validação {perda_validacao:.4f}")

        if perda_validacao < melhor_perda:
            melhor_perda = perda_validacao
            historico["melhor_epoca"] = epoca
            epocas_sem_melhorar = 0
            torch.save(modelo.state_dict(), caminho_pesos)
        else:
            epocas_sem_melhorar += 1
            if epocas_sem_melhorar >= paciencia:
                print(f"    early stopping: {paciencia} épocas sem melhorar")
                break

    historico["tempo_segundos"] = time.time() - inicio
    modelo.load_state_dict(torch.load(caminho_pesos, map_location=dispositivo))  # volta para a melhor época
    return historico


# ---------------------------------------------------------------------------
# Etapa 5 - Avaliação e métricas
# ---------------------------------------------------------------------------

def prever(modelo, carregador):
    """Devolve a probabilidade de pCR de cada imagem do carregador (na mesma ordem)."""
    modelo.eval()
    probabilidades = []
    with torch.no_grad():
        for imagens, _ in carregador:
            logits = modelo(imagens.to(dispositivo)).squeeze(1)
            probabilidades.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilidades)


def agregar_por_paciente(probabilidades, rotulos, pacientes, modo):
    """Junta as previsões dos cortes de cada paciente em uma só (estratégia "todos os cortes").

    modo = "media":  média das probabilidades dos cortes
    modo = "maximo": maior probabilidade entre os cortes
    """
    funcao = np.mean if modo == "media" else np.max
    ids = np.unique(pacientes)
    prob_paciente = np.array([funcao(probabilidades[pacientes == pid]) for pid in ids])
    rotulo_paciente = np.array([rotulos[pacientes == pid][0] for pid in ids])
    return prob_paciente, rotulo_paciente


def calcular_metricas(rotulos, probabilidades, limiar=LIMIAR):
    """Sensibilidade, especificidade, precisão, acurácia e F1, com pCR como classe positiva."""
    previsoes = (probabilidades >= limiar).astype(int)
    vn, fp, fn, vp = confusion_matrix(rotulos.astype(int), previsoes, labels=[0, 1]).ravel()

    def dividir(a, b):
        return a / b if b > 0 else 0.0

    sensibilidade = dividir(vp, vp + fn)
    precisao = dividir(vp, vp + fp)
    return {
        "VP": int(vp), "VN": int(vn), "FP": int(fp), "FN": int(fn),
        "sensibilidade": sensibilidade,
        "especificidade": dividir(vn, vn + fp),
        "precisao": precisao,
        "acuracia": dividir(vp + vn, vp + vn + fp + fn),
        "f1": dividir(2 * precisao * sensibilidade, precisao + sensibilidade),
    }


def salvar_curva_perda(historico, titulo, caminho):
    """Gráfico de convergência: loss de treino e de validação por época."""
    epocas = range(1, len(historico["perda_treino"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epocas, historico["perda_treino"], label="treino")
    plt.plot(epocas, historico["perda_validacao"], label="validação")
    plt.axvline(historico["melhor_epoca"], color="gray", linestyle="--", label="melhor época")
    plt.xlabel("época")
    plt.ylabel("loss (BCE ponderada)")
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho, dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Etapa 6 - Grad-CAM
# ---------------------------------------------------------------------------

def gerar_gradcam(modelo, imagem):
    """Mostra onde a rede "olhou" para decidir.

    imagem: tensor (canais, 256, 256). Retorna (probabilidade de pCR, mapa de calor 256x256 entre 0 e 1).
    Passos: guarda as ativações da última camada convolucional, calcula o gradiente da saída em relação
    a elas, usa a média do gradiente de cada mapa como peso, soma os mapas ponderados e aplica ReLU.
    """
    modelo.eval()
    guardado = {}

    def ao_passar_para_frente(camada, entrada, saida):
        guardado["ativacoes"] = saida
        saida.register_hook(lambda gradiente: guardado.__setitem__("gradientes", gradiente))

    gancho = modelo.camada_gradcam().register_forward_hook(ao_passar_para_frente)
    entrada = imagem.unsqueeze(0).to(dispositivo).requires_grad_(True)  # faz o gradiente passar pela base congelada
    logit = modelo(entrada)
    modelo.zero_grad()
    logit.sum().backward()
    gancho.remove()

    pesos = guardado["gradientes"].mean(dim=(2, 3), keepdim=True)
    mapa = torch.relu((pesos * guardado["ativacoes"]).sum(dim=1, keepdim=True))
    mapa = F.interpolate(mapa, size=imagem.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    mapa = mapa / (mapa.max() + 1e-8)
    return torch.sigmoid(logit).item(), mapa.detach().cpu().numpy()


def salvar_figura_gradcam(imagem, mascara, probabilidade, mapa, rotulo, caminho):
    """Figura com a imagem, o Grad-CAM sobreposto e a máscara do tumor, para comparar as regiões."""
    canal = imagem[-1] if imagem.shape[0] == 1 else imagem[1]  # mostra o pós-contraste precoce
    figura, eixos = plt.subplots(1, 3, figsize=(12, 4))
    eixos[0].imshow(canal, cmap="gray")
    eixos[0].set_title("imagem (pós-contraste precoce)")
    eixos[1].imshow(canal, cmap="gray")
    eixos[1].imshow(mapa, cmap="jet", alpha=0.4)
    eixos[1].set_title(f"Grad-CAM | P(pCR) = {probabilidade:.2f}")
    eixos[2].imshow(mascara, cmap="gray")
    eixos[2].set_title(f"máscara do tumor | pCR real = {int(rotulo)}")
    for eixo in eixos:
        eixo.axis("off")
    plt.tight_layout()
    plt.savefig(caminho, dpi=100)
    plt.close(figura)


# ---------------------------------------------------------------------------
# Etapa 7 - Experimentos: 3 redes x 2 regimes x 2 estratégias
# ---------------------------------------------------------------------------

def separar(dados, particao):
    """Filtra as imagens de uma partição (treino, validação ou teste)."""
    filtro = dados["particoes"] == particao
    return {chave: valor[filtro] for chave, valor in dados.items()}


def executar_experimento(nome_rede, regime, estrategia, dados):
    """Treina uma rede em um regime e estratégia, avalia no teste e salva pesos, gráfico e métricas."""
    fixar_semente()
    nome = f"{nome_rede}_{regime}_{estrategia}"
    canais = REGIMES[regime]
    treino = separar(dados, PARTICAO_TREINO)
    validacao = separar(dados, PARTICAO_VALIDACAO)
    teste = separar(dados, PARTICAO_TESTE)

    # Na estratégia "todos", validar com as 10 variações multiplicaria o tempo por 10; usa a imagem original.
    modo_validacao = "todas" if estrategia == "central" else "nenhum"
    carregador_treino = DataLoader(DatasetPCR(treino["imagens"], treino["rotulos"], canais, "sorteio"),
                                   batch_size=TAMANHO_LOTE, shuffle=True)
    carregador_validacao = DataLoader(DatasetPCR(validacao["imagens"], validacao["rotulos"], canais, modo_validacao),
                                      batch_size=TAMANHO_LOTE)
    carregador_teste = DataLoader(DatasetPCR(teste["imagens"], teste["rotulos"], canais, "nenhum"),
                                  batch_size=TAMANHO_LOTE)

    modelo = REDES[nome_rede](len(canais)).to(dispositivo)
    treinaveis, total = contar_parametros(modelo)
    print(f"\n=== {nome} | {treinaveis:,} parâmetros treináveis de {total:,} | {dispositivo} ===")

    caminho_pesos = os.path.join(PASTA_PESOS, f"{nome}.pt")
    historico = treinar(modelo, carregador_treino, carregador_validacao,
                        calcular_peso_positivo(treino["rotulos"]),
                        EPOCAS[estrategia], PACIENCIA[estrategia], caminho_pesos)
    salvar_curva_perda(historico, nome, os.path.join(PASTA_RESULTADOS, f"loss_{nome}.png"))

    probabilidades = prever(modelo, carregador_teste)
    if estrategia == "central":
        avaliacoes = {"corte central": (probabilidades, teste["rotulos"])}
    else:
        avaliacoes = {
            "média dos cortes": agregar_por_paciente(probabilidades, teste["rotulos"], teste["pacientes"], "media"),
            "máximo dos cortes": agregar_por_paciente(probabilidades, teste["rotulos"], teste["pacientes"], "maximo"),
        }

    linhas = []
    for agregacao, (prob, rot) in avaliacoes.items():
        metricas = calcular_metricas(rot, prob)
        print(f"    teste ({agregacao}): " + ", ".join(
            f"{k} {v:.2f}" for k, v in metricas.items() if isinstance(v, float)))
        linhas.append({
            "rede": nome_rede, "regime": regime, "estrategia": estrategia, "agregacao": agregacao,
            **metricas,
            "melhor_epoca": historico["melhor_epoca"],
            "epocas_rodadas": len(historico["perda_treino"]),
            "tempo_treino_s": round(historico["tempo_segundos"], 1),
            "parametros_treinaveis": treinaveis,
        })
    return linhas, modelo


def salvar_metricas(linhas, caminho):
    """Acrescenta as linhas de métricas a um CSV (cria o cabeçalho se o arquivo não existir)."""
    novo = not os.path.exists(caminho)
    with open(caminho, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0].keys()))
        if novo:
            escritor.writeheader()
        for linha in linhas:
            escritor.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in linha.items()})


def gerar_exemplo_gradcam(modelo, nome, regime, dados):
    """Salva a figura de Grad-CAM do primeiro paciente com pCR do conjunto de teste."""
    teste = separar(dados, PARTICAO_TESTE)
    posicao = int(np.argmax(teste["rotulos"] == 1))
    pid = teste["pacientes"][posicao]
    imagem = torch.from_numpy(teste["imagens"][posicao][REGIMES[regime]].astype(np.float32))
    mascara = ler_mascara(pid)
    mascara = mascara[mascara.shape[0] // 2]
    probabilidade, mapa = gerar_gradcam(modelo, imagem)
    salvar_figura_gradcam(imagem.numpy(), mascara, probabilidade, mapa, teste["rotulos"][posicao],
                          os.path.join(PASTA_RESULTADOS, f"gradcam_{nome}_{pid}.png"))


def main():
    parser = argparse.ArgumentParser(description="Treina e avalia as redes de classificação de pCR.")
    parser.add_argument("--redes", nargs="+", default=list(REDES), choices=list(REDES))
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    parser.add_argument("--estrategias", nargs="+", default=["central", "todos"], choices=["central", "todos"])
    argumentos = parser.parse_args()

    os.makedirs(PASTA_PESOS, exist_ok=True)
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    caminho_metricas = os.path.join(PASTA_RESULTADOS, "metricas.csv")

    for estrategia in argumentos.estrategias:
        print(f"\nCarregando dados (estratégia: {estrategia})...")
        dados = carregar_dados(estrategia)
        print(f"  {len(dados['imagens'])} imagens de {len(np.unique(dados['pacientes']))} pacientes")

        for nome_rede in argumentos.redes:
            for regime in argumentos.regimes:
                linhas, modelo = executar_experimento(nome_rede, regime, estrategia, dados)
                salvar_metricas(linhas, caminho_metricas)
                if estrategia == "central":
                    gerar_exemplo_gradcam(modelo, f"{nome_rede}_{regime}_{estrategia}", regime, dados)

    print(f"\nPronto. Métricas em {caminho_metricas}, pesos em {PASTA_PESOS}")


if __name__ == "__main__":
    main()
