# =============================================================================
# Trabalho Prático - Processamento e Análise de Imagens
# PUC Minas - ICEI - Ciência da Computação - Prof. Alexei Machado
# Classificação da resposta ao tratamento do câncer de mama usando redes neurais
#
# Grupo: Gnn                                   <- PREENCHER o número do grupo
# Componentes (nome completo - matrícula - curso - campus):
#   Bernardo  PREENCHER - PREENCHER - Ciência da Computação - PREENCHER
#   Diego     PREENCHER - PREENCHER - Ciência da Computação - PREENCHER
#   Kauan     PREENCHER - PREENCHER - Ciência da Computação - PREENCHER
#   Túlio     PREENCHER - PREENCHER - Ciência da Computação - PREENCHER
# =============================================================================
#
# O programa classifica a resposta patológica completa (pCR) a partir de DCE-MRI.
#
# Três redes (CNN própria, ResNet50 e VGG16), cada uma treinada em dois regimes:
#   - "1fase":  só o pós-contraste precoce (1 canal)
#   - "3fases": pré-contraste, pós-contraste precoce e pós-contraste tardio (3 canais)
# e com duas estratégias para passar do volume 3D para imagens 2D:
#   - "central": só o corte do meio de cada volume
#   - "todos":   todos os cortes com tumor; a previsão do paciente é a média ou o máximo dos cortes
#
# Como executar (a partir da pasta deste arquivo):
#   python trabalho_pai.py                  abre a interface gráfica
#   python trabalho_pai.py --treinar        treina e avalia as 3 redes x 2 regimes x 2 estratégias
#   python trabalho_pai.py --treinar --redes resnet50 --regimes 3fases --estrategias central
#   python trabalho_pai.py --ajustar        roda as tentativas de ajuste de hiperparâmetros
#
# O dataset fica em "Base de dados/BreastDCEDL_ISPY1_min_crop/" (pastas dce/ e mask/), ao lado deste arquivo.
# A planilha BreastDCEDL_SPY1_min_crop_metadata.csv pode ficar em "Base de dados/" ou ao lado deste arquivo.
#
# Organização do arquivo:
#   Etapa 1 - Leitura dos dados            Etapa 6 - Grad-CAM
#   Etapa 2 - Aumento de dados e Dataset   Etapa 7 - Experimentos
#   Etapa 3 - As três redes                Etapa 8 - Ajuste de hiperparâmetros
#   Etapa 4 - Treino                       Etapa 9 - Interface gráfica
#   Etapa 5 - Avaliação e métricas

import argparse
import contextlib
import os
import queue
import random
import threading
import time
from tkinter import filedialog

import customtkinter as ctk
import matplotlib
matplotlib.use("Agg")  # o pyplot só grava figuras em arquivo; a interface desenha em um canvas próprio
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.ndimage import rotate
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

PASTA_RAIZ = os.path.dirname(os.path.abspath(__file__))
PASTA_BASE = os.path.join(PASTA_RAIZ, "Base de dados")
PASTA_DATASET = os.path.join(PASTA_BASE, "BreastDCEDL_ISPY1_min_crop")
NOME_CSV = "BreastDCEDL_SPY1_min_crop_metadata.csv"
PASTA_PESOS = os.path.join(PASTA_RAIZ, "pesos")          # não vai para o git nem para o ZIP (arquivos grandes)
PASTA_RESULTADOS = os.path.join(PASTA_RAIZ, "resultados")
CAMINHO_METRICAS = os.path.join(PASTA_RESULTADOS, "metricas.csv")
CAMINHO_AJUSTE = os.path.join(PASTA_RESULTADOS, "ajuste.csv")

# Nomes dos arquivos do dataset: <exame>_dce_aqc_<n>.nii.gz (fases) e <exame>_mask.nii.gz (máscara)
MARCA_FASE = "_dce_aqc_"
MARCA_MASCARA = "_mask"
EXTENSOES_NII = (".nii", ".nii.gz")

# As 3 fases de cada exame, na ordem temporal (é a ordem dos canais da imagem de entrada)
FASES = ["pré-contraste", "pós-contraste precoce", "pós-contraste tardio"]

# Coluna "test" do CSV (artigo do BreastDCEDL, Tabela 4): 0 = treino, 1 = teste, 2 = validação
PARTICAO_TREINO, PARTICAO_TESTE, PARTICAO_VALIDACAO = 0, 1, 2
NOMES_PARTICOES = {PARTICAO_TREINO: "treino", PARTICAO_VALIDACAO: "validação", PARTICAO_TESTE: "teste"}

# Canais usados em cada regime (posições dentro de FASES)
REGIMES = {
    "1fase": [1],         # pós-contraste precoce: fase em que o tumor mais realça
    "3fases": [0, 1, 2],  # as três fases como três canais
}

ESTRATEGIAS = ["central", "todos"]

# Aumento de dados exigido no enunciado: 5 ângulos x (original, espelhada) = 10 variações
ANGULOS = [-20, -10, 0, 10, 20]
VARIACOES = [(angulo, espelhar) for angulo in ANGULOS for espelhar in (False, True)]

# Hiperparâmetros do treino final: são os da tentativa T5 do ajuste da Etapa 8, a de menor loss de
# validação (justificativas no relatório).
HIPERPARAMETROS = {
    "taxa_aprendizado": 1e-3,
    "decaimento_pesos": 1e-2,
    "dropout": 0.5,
    "tamanho_lote": 16,
    "semente": 42,
}
LIMIAR_PADRAO = 0.5
EPOCAS = {"central": 40, "todos": 15}     # máximo de épocas
PACIENCIA = {"central": 8, "todos": 4}    # early stopping: épocas sem melhorar a loss de validação

# Ajuste de hiperparâmetros. O ponto de partida são os valores da primeira rodada do grupo (30/10)
# e cada tentativa diz o que muda em relação a ele.
BASE_DO_AJUSTE = {"taxa_aprendizado": 1e-3, "decaimento_pesos": 1e-4, "dropout": 0.5, "tamanho_lote": 16}
TENTATIVAS = {
    # 1ª etapa: taxa de aprendizado
    "T1": {},                              # o próprio ponto de partida
    "T2": {"taxa_aprendizado": 3e-4},
    "T3": {"taxa_aprendizado": 1e-4},
    "T4": {"taxa_aprendizado": 1e-5},
    # 2ª etapa: regularização e tamanho do lote, com a taxa de aprendizado de T1 (a melhor da 1ª etapa)
    "T5": {"decaimento_pesos": 1e-2},
    "T6": {"dropout": 0.7},
    "T7": {"tamanho_lote": 32},
}
SEMENTES_AJUSTE = [42, 43, 44]            # cada tentativa roda com 3 sementes, para medir o ruído

if torch.backends.mps.is_available():
    dispositivo = torch.device("mps")     # GPU dos Macs com chip Apple
elif torch.cuda.is_available():
    dispositivo = torch.device("cuda")    # GPU NVIDIA
else:
    dispositivo = torch.device("cpu")


def fixar_semente(semente):
    """Deixa os sorteios reproduzíveis, para comparar experimentos de forma justa."""
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)


# ---------------------------------------------------------------------------
# Etapa 1 - Leitura dos dados
# ---------------------------------------------------------------------------

def ler_volume(caminho):
    """Lê um arquivo .nii/.nii.gz e devolve o volume (cortes, 256, 256) em float32.

    O eixo 0 do arquivo é a direção z (os cortes); cada volume[k] é um plano xy de 256x256.
    """
    return nib.load(caminho).get_fdata().astype(np.float32)


def normalizar(volume):
    """Normalização z-score por volume: média 0 e desvio padrão 1.

    Cada exame tem uma escala de intensidade diferente (em um paciente o máximo é 110,
    em outro passa de 2000). Sem normalizar, a rede aprenderia a escala do aparelho.
    """
    return (volume - volume.mean()) / (volume.std() + 1e-8)


def ler_fases(caminhos):
    """Lê e normaliza os arquivos das 3 fases e devolve tudo empilhado: (3, cortes, 256, 256)."""
    return np.stack([normalizar(ler_volume(caminho)) for caminho in caminhos])


def localizar_csv(pasta_dataset=PASTA_DATASET):
    """Procura a planilha de metadados perto do dataset. Devolve o caminho ou None."""
    candidatos = [
        os.path.join(os.path.dirname(pasta_dataset), NOME_CSV),  # ao lado da pasta do dataset
        os.path.join(pasta_dataset, NOME_CSV),                   # dentro da pasta do dataset
        os.path.join(PASTA_RAIZ, NOME_CSV),                      # ao lado deste arquivo
    ]
    for caminho in candidatos:
        if os.path.exists(caminho):
            return caminho
    return None


def ler_tabela(pasta_dataset=PASTA_DATASET):
    """Lê a planilha de metadados (uma linha por paciente)."""
    caminho = localizar_csv(pasta_dataset)
    if caminho is None:
        raise FileNotFoundError(f"Planilha {NOME_CSV} não encontrada em {os.path.dirname(pasta_dataset)}")
    return pd.read_csv(caminho)


def indice_da_fase(nome_arquivo):
    """Número N do sufixo _dce_aqc_N do nome de um arquivo de fase."""
    return int(nome_arquivo.split(MARCA_FASE)[1].split(".")[0])


def caminhos_das_fases(pid, pasta_dataset=PASTA_DATASET):
    """Devolve os caminhos dos 3 arquivos de fase de um paciente, em ordem temporal.

    O dataset traz exatamente 3 arquivos por paciente e a ordem do índice N é a ordem temporal
    (pré-contraste, pós-contraste precoce, pós-contraste tardio). Os índices nem sempre são 0/1/2:
    em 31 pacientes são 0/2/3 ou 0/2/5. Usar os arquivos que existem dá o mesmo resultado que as
    colunas pre/post_early/post_late do CSV em 171 pacientes; no ISPY1_1139 o CSV repete o índice 2
    para as duas fases pós-contraste, embora os arquivos sejam 0, 1 e 2.
    """
    pasta = os.path.join(pasta_dataset, "dce")
    arquivos = [nome for nome in os.listdir(pasta) if nome.startswith(f"{pid}_") and MARCA_FASE in nome]
    return [os.path.join(pasta, nome) for nome in sorted(arquivos, key=indice_da_fase)]


def ler_mascara(pid, pasta_dataset=PASTA_DATASET):
    """Devolve a máscara do tumor (cortes, 256, 256), com 1 = tumor e 0 = fundo."""
    return ler_volume(os.path.join(pasta_dataset, "mask", f"{pid}_spy1_vis1{MARCA_MASCARA}.nii.gz"))


def carregar_dados(estrategia, pasta_dataset=PASTA_DATASET):
    """Lê todos os pacientes e devolve as imagens 2D com rótulo, partição e paciente.

    estrategia = "central": 1 imagem por paciente (o corte do meio do volume).
    estrategia = "todos":   1 imagem por corte que contém tumor.

    Retorna um dicionário com:
        imagens   (N, 3, 256, 256)  float16 (metade da memória; vira float32 na hora de usar)
        mascaras  (N, 256, 256)     True onde há tumor (usada para avaliar o Grad-CAM)
        rotulos   (N,)              0 = sem pCR, 1 = com pCR
        particoes (N,)              0 = treino, 1 = teste, 2 = validação
        pacientes (N,)              pid do paciente de cada imagem
    """
    tabela = ler_tabela(pasta_dataset)
    imagens, mascaras, rotulos, particoes, pacientes = [], [], [], [], []

    for numero, linha in tabela.iterrows():
        fases = ler_fases(caminhos_das_fases(linha["pid"], pasta_dataset))
        mascara = ler_mascara(linha["pid"], pasta_dataset) > 0

        if estrategia == "central":
            cortes = [fases.shape[1] // 2]
        else:
            cortes = [k for k in range(mascara.shape[0]) if mascara[k].any()]

        for k in cortes:
            imagens.append(fases[:, k].astype(np.float16))
            mascaras.append(mascara[k])
            rotulos.append(float(linha["pCR"]))
            particoes.append(int(linha["test"]))
            pacientes.append(linha["pid"])

        if (numero + 1) % 43 == 0:
            print(f"  lidos {numero + 1}/{len(tabela)} pacientes")

    return {
        "imagens": np.stack(imagens),
        "mascaras": np.stack(mascaras),
        "rotulos": np.array(rotulos, dtype=np.float32),
        "particoes": np.array(particoes),
        "pacientes": np.array(pacientes),
    }


def localizar_exame(caminho_arquivo):
    """A partir de um arquivo aberto na interface, encontra as fases e a máscara do mesmo exame.

    Funciona para qualquer pasta com a organização do dataset (dce/ e mask/ lado a lado),
    inclusive para pacientes que não estão na planilha.
    Retorna {"nome": prefixo do exame, "fases": caminhos em ordem temporal, "mascara": caminho ou None}.
    """
    pasta, arquivo = os.path.split(os.path.abspath(caminho_arquivo))
    if MARCA_FASE in arquivo:
        nome = arquivo.split(MARCA_FASE)[0]
    elif MARCA_MASCARA in arquivo:
        nome = arquivo.split(MARCA_MASCARA)[0]
    else:
        return {"nome": None, "fases": [], "mascara": None}

    pasta_dataset = os.path.dirname(pasta)
    fases = {}
    mascara = None
    for pasta_candidata in (pasta, os.path.join(pasta_dataset, "dce"), os.path.join(pasta_dataset, "mask")):
        if not os.path.isdir(pasta_candidata):
            continue
        for candidato in os.listdir(pasta_candidata):
            if not candidato.lower().endswith(EXTENSOES_NII):
                continue
            if candidato.startswith(nome + MARCA_FASE):
                fases[indice_da_fase(candidato)] = os.path.join(pasta_candidata, candidato)
            elif candidato.startswith(nome + MARCA_MASCARA):
                mascara = os.path.join(pasta_candidata, candidato)

    # A ordem do índice é a ordem temporal: pré-contraste, pós-contraste precoce e pós-contraste tardio.
    return {"nome": nome, "fases": [fases[indice] for indice in sorted(fases)], "mascara": mascara}


def verificar_desbalanceamento(tabela):
    """Conta os pacientes sem pCR e com pCR em cada partição.

    Retorna uma lista de (nome da partição, sem pCR, com pCR), com o total na última linha.
    No treino são 78 sem pCR para 26 com pCR (3:1): o desbalanceamento é relevante e é tratado
    com peso maior para a classe pCR na função de perda (ver calcular_peso_positivo).
    """
    contagens = []
    for particao, nome in NOMES_PARTICOES.items():
        rotulos = tabela.loc[tabela["test"] == particao, "pCR"]
        contagens.append((nome, int((rotulos == 0).sum()), int((rotulos == 1).sum())))
    contagens.append(("total", int((tabela["pCR"] == 0).sum()), int((tabela["pCR"] == 1).sum())))
    return contagens


# ---------------------------------------------------------------------------
# Etapa 2 - Aumento de dados e Dataset do PyTorch
# ---------------------------------------------------------------------------

def aplicar_variacao(imagem, angulo, espelhar):
    """Aplica uma das 10 variações a uma imagem (canais, 256, 256).

    Todos os canais (fases) recebem exatamente a mesma transformação, para continuarem alinhados.
    Na rotação, reshape=False mantém 256x256, mode="nearest" preenche os cantos com o pixel
    mais próximo e order=1 usa interpolação bilinear.
    """
    if espelhar:
        imagem = imagem[:, :, ::-1]
    if angulo != 0:
        imagem = rotate(imagem, angle=angulo, axes=(2, 1), reshape=False, mode="nearest", order=1)
    return np.ascontiguousarray(imagem)


def gerar_aumentos(corte):
    """Gera as 10 variações de um corte 2D: rotações de -20 a 20 graus, original e espelhada.

    Retorna uma lista de (descrição, imagem), usada na interface para mostrar o aumento de dados.
    """
    variacoes = []
    for angulo, espelhar in VARIACOES:
        descricao = f"{'espelhada' if espelhar else 'original'}, {angulo:+d}°"
        variacoes.append((descricao, aplicar_variacao(corte[np.newaxis], angulo, espelhar)[0]))
    return variacoes


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

    def __init__(self, n_canais, dropout=HIPERPARAMETROS["dropout"], pre_treinada=True):
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
            nn.Dropout(dropout),
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
    pre_treinada=False cria a rede sem baixar os pesos do ImageNet; é usado ao recarregar um
    modelo já treinado, cujo arquivo de pesos contém a rede inteira.
    """

    def __init__(self, n_canais, dropout=HIPERPARAMETROS["dropout"], pre_treinada=True):
        super().__init__()
        self.rede = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pre_treinada else None)
        for parametro in self.rede.parameters():
            parametro.requires_grad = False
        self.rede.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, 1))  # camada nova: treinável

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

    def __init__(self, n_canais, dropout=HIPERPARAMETROS["dropout"], pre_treinada=True):
        super().__init__()
        self.rede = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pre_treinada else None)
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
            nn.Dropout(dropout),
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


def caminho_dos_pesos(nome_rede, regime, estrategia):
    """Arquivo em que ficam os pesos de um modelo treinado."""
    return os.path.join(PASTA_PESOS, f"{nome_rede}_{regime}_{estrategia}.pt")


def salvar_modelo(modelo, caminho, limiares=None):
    """Salva os pesos do modelo e os limiares de decisão escolhidos na validação."""
    torch.save({"pesos": modelo.state_dict(), "limiares": limiares or {}}, caminho)


def carregar_modelo(nome_rede, regime, estrategia):
    """Recria um modelo já treinado a partir do arquivo de pesos, sem treinar de novo.

    Retorna (modelo em modo de avaliação, limiares de decisão escolhidos na validação).
    """
    caminho = caminho_dos_pesos(nome_rede, regime, estrategia)
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Pesos não encontrados: {caminho}. Treine a rede antes de classificar.")
    salvo = torch.load(caminho, map_location=dispositivo)
    modelo = REDES[nome_rede](len(REGIMES[regime]), pre_treinada=False).to(dispositivo)
    modelo.load_state_dict(salvo["pesos"])
    modelo.eval()
    return modelo, salvo["limiares"]


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


def treinar(modelo, carregador_treino, carregador_validacao, peso_positivo, hiper, max_epocas, paciencia,
            caminho_pesos):
    """Treina com early stopping e salva os pesos da época com menor loss de validação.

    Retorna o histórico: loss de treino e de validação por época, melhor época e tempo em segundos.
    """
    funcao_perda = nn.BCEWithLogitsLoss(pos_weight=peso_positivo.to(dispositivo))
    parametros_treinaveis = [p for p in modelo.parameters() if p.requires_grad]
    otimizador = torch.optim.Adam(parametros_treinaveis, lr=hiper["taxa_aprendizado"],
                                  weight_decay=hiper["decaimento_pesos"])

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
            salvar_modelo(modelo, caminho_pesos)
        else:
            epocas_sem_melhorar += 1
            if epocas_sem_melhorar >= paciencia:
                print(f"    early stopping: {paciencia} épocas sem melhorar")
                break

    historico["tempo_segundos"] = time.time() - inicio
    historico["melhor_perda_validacao"] = melhor_perda
    modelo.load_state_dict(torch.load(caminho_pesos, map_location=dispositivo)["pesos"])  # volta para a melhor época
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


def calcular_metricas(rotulos, probabilidades, limiar=LIMIAR_PADRAO):
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


def calcular_auc(rotulos, probabilidades):
    """Área sob a curva ROC: mede a separação entre as classes sem depender do limiar (0,5 = acaso)."""
    if len(np.unique(rotulos)) < 2:
        return float("nan")
    return float(roc_auc_score(rotulos, probabilidades))


def escolher_limiar(rotulos, probabilidades):
    """Escolhe o limiar de decisão que maximiza o índice de Youden (sensibilidade + especificidade - 1).

    Deve ser chamada com as previsões da VALIDAÇÃO; o limiar escolhido é depois aplicado ao teste.
    O índice de Youden dá o mesmo peso às duas classes. Maximizar o F1 favoreceria o limiar que
    classifica quase todos como pCR, por causa do desbalanceamento.
    Os candidatos são os pontos médios entre probabilidades vizinhas; em empate, vence o mais perto de 0,5.
    """
    ordenadas = np.unique(probabilidades)
    candidatos = np.append((ordenadas[:-1] + ordenadas[1:]) / 2, LIMIAR_PADRAO)
    melhor_limiar, melhor_indice = LIMIAR_PADRAO, -1.0
    for limiar in candidatos:
        metricas = calcular_metricas(rotulos, probabilidades, limiar)
        indice = metricas["sensibilidade"] + metricas["especificidade"] - 1
        empate = abs(indice - melhor_indice) < 1e-9
        mais_central = abs(limiar - LIMIAR_PADRAO) < abs(melhor_limiar - LIMIAR_PADRAO)
        if indice > melhor_indice + 1e-9 or (empate and mais_central):
            melhor_limiar, melhor_indice = float(limiar), indice
    return melhor_limiar


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


def desenhar_gradcam(eixo, corte, mapa, mascara=None):
    """Desenha o corte com o mapa de calor por cima e, se houver máscara, o contorno do tumor em branco."""
    eixo.imshow(corte, cmap="gray")
    eixo.imshow(mapa, cmap="jet", alpha=0.4)
    if mascara is not None and mascara.any():
        eixo.contour(mascara, levels=[0.5], colors="white", linewidths=1)
    eixo.axis("off")


def salvar_figura_gradcam(imagem, mascara, probabilidade, mapa, rotulo, titulo, caminho):
    """Figura com a imagem, o Grad-CAM sobreposto e a máscara do tumor, para comparar as regiões."""
    canal = imagem[-1] if imagem.shape[0] == 1 else imagem[1]  # mostra o pós-contraste precoce
    figura, eixos = plt.subplots(1, 3, figsize=(9, 3.2))
    eixos[0].imshow(canal, cmap="gray")
    eixos[0].set_title("pós-contraste precoce", fontsize=10)
    eixos[0].axis("off")
    desenhar_gradcam(eixos[1], canal, mapa, mascara)
    eixos[1].set_title(f"Grad-CAM | P(pCR) = {probabilidade:.2f}", fontsize=10)
    eixos[2].imshow(mascara, cmap="gray")
    eixos[2].set_title(f"máscara do tumor | pCR real = {int(rotulo)}", fontsize=10)
    eixos[2].axis("off")
    figura.suptitle(titulo, fontsize=10)
    plt.tight_layout()
    plt.savefig(caminho, dpi=100)
    plt.close(figura)


def medir_gradcam_no_tumor(modelo, imagens, mascaras, canais):
    """Mede, em um conjunto de imagens, se o Grad-CAM aponta para a região do tumor.

    Retorna um dicionário com:
        gradcam_pico_no_tumor:  fração das imagens em que o ponto mais quente do mapa cai dentro da máscara
        gradcam_calor_no_tumor: fração média do calor do mapa que fica dentro da máscara
        area_do_tumor:          fração média da imagem ocupada pelo tumor. É o valor que
                                gradcam_calor_no_tumor teria se o mapa fosse espalhado ao acaso.
    """
    picos, calores, areas = [], [], []
    for imagem, mascara in zip(imagens, mascaras):
        if not mascara.any():
            continue
        _, mapa = gerar_gradcam(modelo, torch.from_numpy(imagem[canais].astype(np.float32)))
        if mapa.sum() <= 0:   # mapa vazio: nenhuma região aumentou a saída de pCR
            continue
        linha, coluna = np.unravel_index(np.argmax(mapa), mapa.shape)
        picos.append(float(mascara[linha, coluna]))
        calores.append(float(mapa[mascara].sum() / mapa.sum()))
        areas.append(float(mascara.mean()))

    def media(valores):
        return float(np.mean(valores)) if valores else float("nan")

    return {"gradcam_pico_no_tumor": media(picos), "gradcam_calor_no_tumor": media(calores),
            "area_do_tumor": media(areas)}


# ---------------------------------------------------------------------------
# Etapa 7 - Experimentos: 3 redes x 2 regimes x 2 estratégias
# ---------------------------------------------------------------------------

def separar(dados, particao):
    """Filtra as imagens de uma partição (treino, validação ou teste)."""
    filtro = dados["particoes"] == particao
    return {chave: valor[filtro] for chave, valor in dados.items()}


def executar_experimento(nome_rede, regime, estrategia, dados, hiper=None, tentativa=""):
    """Treina uma rede em um regime e estratégia, escolhe o limiar na validação e avalia no teste.

    hiper:     hiperparâmetros que mudam em relação a HIPERPARAMETROS (dicionário).
    tentativa: nome da tentativa de ajuste. Vazio no treino final, que guarda os pesos e a curva de loss.
    Retorna (linhas de métricas, modelo treinado, histórico do treino, limiares escolhidos).
    """
    hiper = {**HIPERPARAMETROS, **(hiper or {})}
    fixar_semente(hiper["semente"])
    nome = f"{nome_rede}_{regime}_{estrategia}"
    canais = REGIMES[regime]
    treino = separar(dados, PARTICAO_TREINO)
    validacao = separar(dados, PARTICAO_VALIDACAO)
    teste = separar(dados, PARTICAO_TESTE)

    def criar_carregador(parte, modo_aumento, embaralhar=False):
        return DataLoader(DatasetPCR(parte["imagens"], parte["rotulos"], canais, modo_aumento),
                          batch_size=hiper["tamanho_lote"], shuffle=embaralhar)

    modelo = REDES[nome_rede](len(canais), hiper["dropout"]).to(dispositivo)
    treinaveis, total = contar_parametros(modelo)
    print(f"\n=== {nome} {tentativa} | {treinaveis:,} parâmetros treináveis de {total:,} | {dispositivo} ===")
    print(f"    hiperparâmetros: {hiper}")

    # Na estratégia "todos", validar com as 10 variações multiplicaria o tempo por 10; usa a imagem original.
    modo_validacao = "todas" if estrategia == "central" else "nenhum"
    final = tentativa == ""
    caminho_pesos = caminho_dos_pesos(nome_rede, regime, estrategia)
    if not final:
        caminho_pesos = os.path.join(PASTA_PESOS, "_tentativa.pt")   # as tentativas de ajuste não guardam pesos
    historico = treinar(modelo, criar_carregador(treino, "sorteio", embaralhar=True),
                        criar_carregador(validacao, modo_validacao),
                        calcular_peso_positivo(treino["rotulos"]), hiper,
                        EPOCAS[estrategia], PACIENCIA[estrategia], caminho_pesos)

    # Previsões sem aumento de dados: a validação escolhe o limiar e o teste mede o resultado final.
    prob_validacao = prever(modelo, criar_carregador(validacao, "nenhum"))
    prob_teste = prever(modelo, criar_carregador(teste, "nenhum"))
    if estrategia == "central":
        avaliacoes = {"corte central": ((prob_validacao, validacao["rotulos"]), (prob_teste, teste["rotulos"]))}
    else:
        avaliacoes = {
            descricao: (agregar_por_paciente(prob_validacao, validacao["rotulos"], validacao["pacientes"], modo),
                        agregar_por_paciente(prob_teste, teste["rotulos"], teste["pacientes"], modo))
            for descricao, modo in (("média dos cortes", "media"), ("máximo dos cortes", "maximo"))
        }

    gradcam = {}
    if final:
        gradcam = medir_gradcam_no_tumor(modelo, teste["imagens"], teste["mascaras"], canais)

    linhas, limiares = [], {}
    for agregacao, ((prob_v, rotulos_v), (prob_t, rotulos_t)) in avaliacoes.items():
        limiares[agregacao] = escolher_limiar(rotulos_v, prob_v)
        for criterio, limiar in (("fixo em 0,5", LIMIAR_PADRAO), ("escolhido na validação", limiares[agregacao])):
            metricas = calcular_metricas(rotulos_t, prob_t, limiar)
            print(f"    teste ({agregacao}, limiar {limiar:.2f}): " + ", ".join(
                f"{k} {v:.2f}" for k, v in metricas.items() if isinstance(v, float)))
            linhas.append({
                "tentativa": tentativa, "rede": nome_rede, "regime": regime, "estrategia": estrategia,
                "agregacao": agregacao, "criterio_limiar": criterio, "limiar": limiar,
                **metricas,
                "auc_teste": calcular_auc(rotulos_t, prob_t),
                "auc_validacao": calcular_auc(rotulos_v, prob_v),
                "perda_validacao": historico["melhor_perda_validacao"],
                "melhor_epoca": historico["melhor_epoca"],
                "epocas_rodadas": len(historico["perda_treino"]),
                "tempo_treino_s": round(historico["tempo_segundos"], 1),
                "parametros_treinaveis": treinaveis,
                **hiper,
                **gradcam,
            })

    if final:
        salvar_modelo(modelo, caminho_pesos, limiares)  # regrava os pesos, agora com os limiares
        salvar_curva_perda(historico, nome, os.path.join(PASTA_RESULTADOS, f"loss_{nome}.png"))
    return linhas, modelo, historico, limiares


def salvar_metricas(linhas, caminho):
    """Grava as linhas de métricas em um CSV. Se o mesmo experimento já estiver lá, ele é substituído."""
    novas = pd.DataFrame(linhas)
    if os.path.exists(caminho):
        antigas = pd.read_csv(caminho)
        repetida = pd.Series(True, index=antigas.index)
        for coluna in ("tentativa", "rede", "regime", "estrategia", "semente"):
            repetida &= antigas[coluna].fillna("").astype(str) == str(linhas[0][coluna])
        novas = pd.concat([antigas[~repetida], novas], ignore_index=True)
    novas.round(4).to_csv(caminho, index=False)


def gerar_exemplos_gradcam(modelo, nome, regime, dados, limiar):
    """Salva figuras de Grad-CAM com um exemplo de cada tipo de acerto e de erro no conjunto de teste.

    VP = pCR previsto e real; VN = sem pCR previsto e real; FP = pCR previsto, mas real sem pCR;
    FN = sem pCR previsto, mas real com pCR. Usa o primeiro paciente de cada tipo.
    """
    teste = separar(dados, PARTICAO_TESTE)
    canais = REGIMES[regime]
    faltam = {"VP", "VN", "FP", "FN"}
    for posicao in range(len(teste["rotulos"])):
        imagem = torch.from_numpy(teste["imagens"][posicao][canais].astype(np.float32))
        probabilidade, mapa = gerar_gradcam(modelo, imagem)
        rotulo = int(teste["rotulos"][posicao])
        previsto = int(probabilidade >= limiar)
        tipo = ("V" if previsto == rotulo else "F") + ("P" if previsto == 1 else "N")
        if tipo in faltam:
            faltam.remove(tipo)
            pid = teste["pacientes"][posicao]
            salvar_figura_gradcam(imagem.numpy(), teste["mascaras"][posicao], probabilidade, mapa, rotulo,
                                  f"{nome} | {pid} | {tipo} (limiar {limiar:.2f})",
                                  os.path.join(PASTA_RESULTADOS, f"gradcam_{nome}_{tipo}.png"))
        if not faltam:
            break


def salvar_grade_de_curvas(historicos, redes, regimes, estrategia, caminho):
    """Uma figura só com as curvas de loss dos modelos de uma estratégia: redes nas colunas, regimes nas linhas."""
    figura, eixos = plt.subplots(len(regimes), len(redes), figsize=(4 * len(redes), 2.8 * len(regimes)),
                                 squeeze=False)
    for linha, regime in enumerate(regimes):
        for coluna, nome_rede in enumerate(redes):
            historico = historicos[(nome_rede, regime)]
            eixo = eixos[linha][coluna]
            epocas = range(1, len(historico["perda_treino"]) + 1)
            eixo.plot(epocas, historico["perda_treino"], label="treino")
            eixo.plot(epocas, historico["perda_validacao"], label="validação")
            eixo.axvline(historico["melhor_epoca"], color="gray", linestyle="--", label="melhor época")
            eixo.set_title(f"{nome_rede} | {regime} | {estrategia}", fontsize=10)
            eixo.set_xlabel("época")
            eixo.set_ylabel("loss")
    eixos[0][0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close(figura)


def treinar_redes(redes, regimes, estrategias, hiper=None):
    """Treino final: roda todas as combinações pedidas e salva pesos, métricas, curvas e Grad-CAM."""
    os.makedirs(PASTA_PESOS, exist_ok=True)
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)

    for estrategia in estrategias:
        print(f"\nCarregando dados (estratégia: {estrategia})...")
        dados = carregar_dados(estrategia)
        print(f"  {len(dados['imagens'])} imagens de {len(np.unique(dados['pacientes']))} pacientes")

        historicos = {}
        for nome_rede in redes:
            for regime in regimes:
                linhas, modelo, historico, limiares = executar_experimento(nome_rede, regime, estrategia, dados, hiper)
                salvar_metricas(linhas, CAMINHO_METRICAS)
                historicos[(nome_rede, regime)] = historico
                if estrategia == "central":
                    gerar_exemplos_gradcam(modelo, f"{nome_rede}_{regime}_{estrategia}", regime, dados,
                                           limiares["corte central"])

        if len(historicos) > 1:   # com um modelo só, a curva individual loss_<nome>.png já basta
            salvar_grade_de_curvas(historicos, redes, regimes, estrategia,
                                   os.path.join(PASTA_RESULTADOS, f"convergencia_{estrategia}.png"))

    print(f"\nPronto. Métricas em {CAMINHO_METRICAS}, pesos em {PASTA_PESOS}")


# ---------------------------------------------------------------------------
# Etapa 8 - Ajuste de hiperparâmetros
# ---------------------------------------------------------------------------

def salvar_comparacao_de_convergencia(historicos, redes, regimes, caminho):
    """Uma figura com a loss de validação (linha cheia) e de treino (tracejada) de cada tentativa."""
    figura, eixos = plt.subplots(len(regimes), len(redes), figsize=(4 * len(redes), 2.8 * len(regimes)),
                                 squeeze=False, sharex=True)
    for linha, regime in enumerate(regimes):
        for coluna, nome_rede in enumerate(redes):
            eixo = eixos[linha][coluna]
            for numero, tentativa in enumerate(TENTATIVAS):
                historico = historicos[(nome_rede, regime, tentativa)]
                epocas = range(1, len(historico["perda_treino"]) + 1)
                cor = f"C{numero}"
                eixo.plot(epocas, historico["perda_validacao"], color=cor, label=tentativa)
                eixo.plot(epocas, historico["perda_treino"], color=cor, linestyle="--", alpha=0.6)
            eixo.set_title(f"{nome_rede} | {regime}", fontsize=10)
    for linha_de_eixos in eixos:
        linha_de_eixos[0].set_ylabel("loss")
    for eixo in eixos[-1]:
        eixo.set_xlabel("época")
    eixos[0][0].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close(figura)


def resumir_ajuste():
    """Resume o CSV do ajuste: média e desvio padrão, entre as sementes, de cada tentativa e rede."""
    tabela = pd.read_csv(CAMINHO_AJUSTE)
    tabela = tabela[tabela["criterio_limiar"] == "escolhido na validação"]
    resumo = tabela.groupby(["tentativa", "rede", "regime"]).agg(
        auc_validacao=("auc_validacao", "mean"), auc_validacao_dp=("auc_validacao", "std"),
        perda_validacao=("perda_validacao", "mean"),
        auc_teste=("auc_teste", "mean"), f1_teste=("f1", "mean"),
        melhor_epoca=("melhor_epoca", "mean"), tempo_treino_s=("tempo_treino_s", "mean"),
    ).round(3).reset_index()
    caminho = os.path.join(PASTA_RESULTADOS, "ajuste_resumo.csv")
    resumo.to_csv(caminho, index=False)
    print(f"\nResumo do ajuste (média das sementes {SEMENTES_AJUSTE}):")
    print(resumo.to_string(index=False))
    print(f"\nResumo salvo em {caminho}")


def ajustar_hiperparametros(redes, regimes):
    """Roda todas as TENTATIVAS com várias sementes e registra tudo em resultados/ajuste.csv.

    Usa a estratégia do corte central, que treina em segundos e permite repetir cada tentativa.
    A escolha dos hiperparâmetros olha só para a VALIDAÇÃO (AUC e loss); o teste fica para o treino final.
    """
    os.makedirs(PASTA_PESOS, exist_ok=True)
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    if os.path.exists(CAMINHO_AJUSTE):
        os.remove(CAMINHO_AJUSTE)

    print("\nCarregando dados (estratégia: central)...")
    dados = carregar_dados("central")
    historicos = {}
    for tentativa, mudancas in TENTATIVAS.items():
        for semente in SEMENTES_AJUSTE:
            for nome_rede in redes:
                for regime in regimes:
                    hiper = {**BASE_DO_AJUSTE, **mudancas, "semente": semente}
                    linhas, _, historico, _ = executar_experimento(nome_rede, regime, "central", dados, hiper,
                                                                   tentativa)
                    salvar_metricas(linhas, CAMINHO_AJUSTE)
                    historicos.setdefault((nome_rede, regime, tentativa), historico)  # guarda a 1ª semente

    salvar_comparacao_de_convergencia(historicos, redes, regimes,
                                      os.path.join(PASTA_RESULTADOS, "ajuste_convergencia.png"))
    resumir_ajuste()


# ---------------------------------------------------------------------------
# Etapa 9 - Interface gráfica
# ---------------------------------------------------------------------------

# Texto mostrado nos menus da interface -> nome usado no código
OPCOES_REDE = {"ResNet50": "resnet50", "CNN própria": "cnn", "VGG16": "vgg16"}
OPCOES_REGIME = {"3 fases": "3fases", "1 fase (pós-contraste precoce)": "1fase"}
OPCOES_ESTRATEGIA = {"Corte central": "central", "Todos os cortes (média)": "todos"}

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("green")


class VisualizadorNii:
    """Mostra um volume NIfTI corte a corte no plano xy, com a máscara do tumor ao lado.

    O terceiro painel recebe o Grad-CAM depois que a imagem é classificada.
    """

    def __init__(self, frame_pai):
        self.frame_pai = frame_pai

        self.volume = None       # (cortes, 256, 256): o eixo 0 é a direção z
        self.mascara = None      # mesmo formato do volume, ou None se o exame não tiver máscara
        self.exame = {"nome": None, "fases": [], "mascara": None}
        self.nome_arquivo = ""
        self.corte_atual = 0
        self.ao_mudar_corte = None   # função avisada quando o corte muda (a interface limpa o resultado)

        self.figura = None
        self.eixos = None
        self.imagem_plot = None
        self.mascara_plot = None
        self.canvas_plot = None
        self.slider_corte = None
        self.label_corte = None
        self.cor_texto = "black"

        self.mostrar_mensagem_inicial()

    def limpar_frame(self):
        """Remove os widgets atuais da área de visualização."""
        for widget in self.frame_pai.winfo_children():
            widget.destroy()

    def mostrar_mensagem_inicial(self):
        self.limpar_frame()
        ctk.CTkLabel(
            self.frame_pai,
            text="Selecione um arquivo .nii para visualizar",
            font=("Arial", 16),
        ).pack(expand=True)

    def carregar_arquivo(self, caminho_arquivo):
        """Carrega um volume .nii, procura a máscara do mesmo exame e monta a visualização."""
        try:
            volume = ler_volume(caminho_arquivo)
            if volume.ndim != 3:
                raise ValueError(f"Era esperado um volume 3D, mas o arquivo possui {volume.ndim} dimensões.")

            exame = localizar_exame(caminho_arquivo)
            caminho_arquivo = os.path.abspath(caminho_arquivo)
            mascara = None
            if exame["mascara"] and exame["mascara"] != caminho_arquivo:
                mascara = ler_volume(exame["mascara"]) > 0
                if mascara.shape != volume.shape:
                    mascara = None

            self.volume, self.mascara, self.exame = volume, mascara, exame
            self.nome_arquivo = os.path.basename(caminho_arquivo)
            if caminho_arquivo in exame["fases"][:3]:   # acrescenta ao título qual das 3 fases foi aberta
                self.nome_arquivo += f"\n{FASES[exame['fases'].index(caminho_arquivo)]}"
            self.corte_atual = volume.shape[0] // 2
            self.montar_visualizacao()

        except Exception as erro:
            self.volume = None
            self.limpar_frame()
            ctk.CTkLabel(
                self.frame_pai,
                text=f"Erro ao abrir NIfTI:\n{erro}",
                text_color="red",
                wraplength=600,
            ).pack(expand=True, padx=20, pady=20)

    def montar_visualizacao(self):
        """Cria a figura com os três painéis e o controle deslizante de cortes."""
        self.limpar_frame()

        escuro = ctk.get_appearance_mode() == "Dark"
        cor_fundo = "#2b2b2b" if escuro else "#f0f0f0"
        self.cor_texto = "white" if escuro else "black"

        # Mesma janela de brilho em todos os cortes: do mínimo até o percentil 99,5 do volume.
        minimo = float(self.volume.min())
        maximo = float(np.percentile(self.volume, 99.5))
        if maximo <= minimo:
            maximo = float(self.volume.max()) if self.volume.max() > minimo else minimo + 1

        # layout="constrained" reajusta os três painéis sempre que a janela muda de tamanho
        self.figura = Figure(figsize=(10, 4), facecolor=cor_fundo, layout="constrained")
        self.eixos = self.figura.subplots(1, 3)
        for eixo in self.eixos:
            eixo.axis("off")

        self.imagem_plot = self.eixos[0].imshow(self.volume[self.corte_atual], cmap="gray", vmin=minimo, vmax=maximo)
        self.eixos[0].set_title(self.nome_arquivo, color=self.cor_texto, fontsize=9)

        if self.mascara is not None:
            self.mascara_plot = self.eixos[1].imshow(self.mascara[self.corte_atual], cmap="gray", vmin=0, vmax=1)
            self.eixos[1].set_title("Segmentação do tumor", color=self.cor_texto, fontsize=9)
        else:
            self.mascara_plot = None
            aviso = "Segmentação não encontrada"
            if self.exame["mascara"]:
                aviso = "O arquivo aberto já é a segmentação"
            self.eixos[1].set_title(aviso, color=self.cor_texto, fontsize=9)

        self.limpar_gradcam()

        self.canvas_plot = FigureCanvasTkAgg(self.figura, master=self.frame_pai)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        total_cortes = self.volume.shape[0]
        self.label_corte = ctk.CTkLabel(
            self.frame_pai,
            text=f"Corte z: {self.corte_atual} / {total_cortes - 1}",
            font=("Arial", 14),
        )
        self.label_corte.pack(pady=5)

        self.slider_corte = ctk.CTkSlider(
            self.frame_pai,
            from_=0,
            to=max(total_cortes - 1, 1),
            number_of_steps=max(total_cortes - 1, 1),
            command=self.atualizar_corte,
        )
        self.slider_corte.set(self.corte_atual)
        self.slider_corte.pack(fill="x", padx=30, pady=10)

        self.canvas_plot.draw()

    def atualizar_corte(self, valor):
        """Troca o corte z da imagem e da máscara quando o usuário move o controle deslizante."""
        if self.volume is None:
            return

        total_cortes = self.volume.shape[0]
        self.corte_atual = min(int(round(float(valor))), total_cortes - 1)
        self.imagem_plot.set_data(self.volume[self.corte_atual])
        if self.mascara_plot is not None:
            self.mascara_plot.set_data(self.mascara[self.corte_atual])
        self.limpar_gradcam()   # o mapa mostrado era de outro corte

        self.label_corte.configure(text=f"Corte z: {self.corte_atual} / {total_cortes - 1}")
        self.canvas_plot.draw_idle()
        if self.ao_mudar_corte is not None:
            self.ao_mudar_corte()

    def limpar_gradcam(self):
        """Deixa o terceiro painel vazio, à espera de uma classificação."""
        eixo = self.eixos[2]
        eixo.clear()
        eixo.axis("off")
        eixo.set_title("Grad-CAM (após classificar)", color=self.cor_texto, fontsize=9)

    def mostrar_gradcam(self, mapa, titulo):
        """Mostra no terceiro painel o corte atual com o mapa de calor e o contorno do tumor."""
        eixo = self.eixos[2]
        eixo.clear()
        mascara = self.mascara[self.corte_atual] if self.mascara is not None else None
        desenhar_gradcam(eixo, self.volume[self.corte_atual], mapa, mascara)
        eixo.set_title(titulo, color=self.cor_texto, fontsize=9)
        self.canvas_plot.draw_idle()


class EscritorDeFila:
    """Recebe o texto dos print() de uma thread e guarda em uma fila, para a interface mostrar."""

    def __init__(self, fila):
        self.fila = fila

    def write(self, texto):
        self.fila.put(texto)

    def flush(self):
        pass


class InterfaceMRI:
    """Define a janela e os controles principais da aplicação."""

    def __init__(self, root):
        self.root = root
        self.root.geometry("1280x720")   # cabe em telas de notebook de 1366x768
        self.root.title("Classificação de pCR em DCE-MRI")

        self.modelos = {}          # modelos já carregados: (rede, regime, estrategia) -> (modelo, limiares)
        self.fases_do_exame = (None, None)   # (nome do exame, fases normalizadas) do último exame classificado
        self.fila_log = queue.Queue()
        self.caixa_log = None
        self.treinando = False

        self.criar_layout()

    # ----- montagem da janela -----

    def criar_layout(self):
        self.frame_esquerdo = ctk.CTkFrame(self.root, width=320)
        self.frame_esquerdo.pack(side="left", fill="y", padx=15, pady=10)
        self.frame_esquerdo.pack_propagate(False)

        ctk.CTkButton(
            self.frame_esquerdo,
            text="Selecionar pasta MRI",
            command=self.selecionar_pasta,
        ).pack(pady=(20, 10), padx=10, fill="x")

        self.lista_arquivos = ctk.CTkScrollableFrame(self.frame_esquerdo, label_text="Arquivos disponíveis")
        self.lista_arquivos.pack(pady=10, padx=10, fill="both", expand=True)

        # Escolha do modelo usado para classificar (e para treinar pela interface)
        self.menu_rede = self.criar_menu(OPCOES_REDE)
        self.menu_regime = self.criar_menu(OPCOES_REGIME)
        self.menu_estrategia = self.criar_menu(OPCOES_ESTRATEGIA)

        for texto, comando in (
            ("Classificar + Grad-CAM", self.classificar),
            ("Aumento de dados (10 variações)", self.mostrar_aumento_de_dados),
            ("Desbalanceamento das classes", self.mostrar_desbalanceamento),
            ("Treinar rede selecionada", self.treinar_rede_selecionada),
        ):
            ctk.CTkButton(self.frame_esquerdo, text=texto, command=comando).pack(pady=4, padx=10, fill="x")

        frame_direito = ctk.CTkFrame(self.root, fg_color="transparent")
        frame_direito.pack(side="right", fill="both", expand=True, padx=15, pady=10)

        self.label_resultado = ctk.CTkLabel(
            frame_direito,
            text="Abra uma imagem e clique em \"Classificar + Grad-CAM\".",
            font=("Arial", 15),
            justify="left",
            anchor="w",
            wraplength=880,
        )
        self.label_resultado.pack(side="bottom", fill="x", padx=10, pady=(5, 10))

        self.frame_visualizador = ctk.CTkFrame(frame_direito)
        self.frame_visualizador.pack(side="top", fill="both", expand=True)

        # A interface delega a leitura do NIfTI e o desenho ao visualizador.
        self.visualizador = VisualizadorNii(self.frame_visualizador)
        self.visualizador.ao_mudar_corte = lambda: self.mostrar_resultado(
            "Corte alterado. Clique em \"Classificar + Grad-CAM\" para classificar este corte.")

    def criar_menu(self, opcoes):
        menu = ctk.CTkOptionMenu(self.frame_esquerdo, values=list(opcoes))
        menu.pack(pady=3, padx=10, fill="x")
        return menu

    def mostrar_resultado(self, texto, cor=None):
        self.label_resultado.configure(text=texto, text_color=cor or ("black", "white"))

    def modelo_selecionado(self):
        """Devolve (rede, regime, estratégia) escolhidos nos menus, com os nomes usados no código."""
        return (OPCOES_REDE[self.menu_rede.get()], OPCOES_REGIME[self.menu_regime.get()],
                OPCOES_ESTRATEGIA[self.menu_estrategia.get()])

    # ----- a) leitura e visualização -----

    def selecionar_pasta(self):
        pasta = filedialog.askdirectory(title="Selecione a pasta das imagens")
        if pasta:
            self.listar_arquivos(pasta)

    def listar_arquivos(self, pasta):
        """Lista os arquivos .nii da pasta. Se for a pasta do dataset, lista o que está em dce/."""
        if os.path.isdir(os.path.join(pasta, "dce")):
            pasta = os.path.join(pasta, "dce")

        for widget in self.lista_arquivos.winfo_children():
            widget.destroy()

        try:
            arquivos = sorted(
                nome
                for nome in os.listdir(pasta)
                if os.path.isfile(os.path.join(pasta, nome)) and nome.lower().endswith(EXTENSOES_NII)
            )

            if not arquivos:
                ctk.CTkLabel(
                    self.lista_arquivos,
                    text="Nenhum arquivo .nii ou .nii.gz encontrado.",
                    wraplength=220,
                ).pack(pady=20, padx=10)
                return

            for nome_arquivo in arquivos:
                caminho = os.path.join(pasta, nome_arquivo)
                ctk.CTkButton(
                    self.lista_arquivos,
                    text=nome_arquivo,
                    anchor="w",
                    fg_color="transparent",
                    text_color=("black", "white"),
                    hover_color=("gray85", "gray25"),
                    command=lambda caminho=caminho: self.abrir_arquivo(caminho),
                ).pack(fill="x", padx=10, pady=2)

        except OSError as erro:
            ctk.CTkLabel(
                self.lista_arquivos,
                text=f"Erro ao ler a pasta:\n{erro}",
                text_color="red",
                wraplength=220,
            ).pack(pady=20, padx=10)

    def abrir_arquivo(self, caminho):
        self.visualizador.carregar_arquivo(caminho)
        self.mostrar_resultado("Imagem aberta. Clique em \"Classificar + Grad-CAM\" para classificar o corte mostrado.")

    # ----- e) classificação e Grad-CAM -----

    def obter_modelo(self, nome_rede, regime, estrategia):
        """Carrega o modelo treinado do disco na primeira vez e reaproveita nas seguintes."""
        chave = (nome_rede, regime, estrategia)
        if chave not in self.modelos:
            self.modelos[chave] = carregar_modelo(nome_rede, regime, estrategia)
        return self.modelos[chave]

    def obter_fases(self, exame):
        """Lê e normaliza as 3 fases do exame aberto (guarda o último exame para não reler a cada clique)."""
        if self.fases_do_exame[0] != exame["nome"]:
            self.fases_do_exame = (exame["nome"], ler_fases(exame["fases"][:3]))
        return self.fases_do_exame[1]

    def rotulo_real(self, exame):
        """Procura na planilha o pCR real do paciente. Devolve 0, 1 ou None se não encontrar."""
        try:
            pasta_dataset = os.path.dirname(os.path.dirname(exame["fases"][0]))
            tabela = ler_tabela(pasta_dataset)
            linha = tabela[[exame["nome"].startswith(f"{pid}_") for pid in tabela["pid"]]]
            return int(linha["pCR"].iloc[0]) if len(linha) else None
        except (FileNotFoundError, KeyError, ValueError):
            return None

    def classificar(self):
        """Classifica a imagem aberta com a rede escolhida e mostra a probabilidade e o Grad-CAM."""
        visualizador = self.visualizador
        exame = visualizador.exame
        if visualizador.volume is None or len(exame["fases"]) < 3:
            self.mostrar_resultado("Abra antes uma imagem de DCE-MRI de um exame com as 3 fases (pasta dce/).", "red")
            return

        nome_rede, regime, estrategia = self.modelo_selecionado()
        try:
            modelo, limiares = self.obter_modelo(nome_rede, regime, estrategia)
            fases = self.obter_fases(exame)
        except Exception as erro:
            self.mostrar_resultado(f"Não foi possível classificar: {erro}", "red")
            return

        canais = REGIMES[regime]
        corte = visualizador.corte_atual
        prob_do_corte, mapa = gerar_gradcam(modelo, torch.from_numpy(fases[:, corte][canais]))
        linhas = [f"{self.menu_rede.get()} | {self.menu_regime.get()} | {self.menu_estrategia.get()}",
                  f"Corte {corte}: P(pCR) = {prob_do_corte:.3f}"]

        if estrategia == "central":
            probabilidade = prob_do_corte
            limiar = limiares.get("corte central", LIMIAR_PADRAO)
        else:
            # Probabilidade do paciente: média das previsões de todos os cortes com tumor.
            # Sem máscara (ou com máscara vazia), usa todos os cortes do volume.
            mascara = visualizador.mascara
            cortes = [k for k in range(fases.shape[1]) if mascara is not None and mascara[k].any()]
            cortes = cortes or list(range(fases.shape[1]))
            imagens = fases[:, cortes].transpose(1, 0, 2, 3)   # (cortes, fases, 256, 256)
            carregador = DataLoader(DatasetPCR(imagens, np.zeros(len(cortes), dtype=np.float32), canais, "nenhum"),
                                    batch_size=HIPERPARAMETROS["tamanho_lote"])
            probabilidade = float(prever(modelo, carregador).mean())
            limiar = limiares.get("média dos cortes", LIMIAR_PADRAO)
            linhas.append(f"Paciente (média de {len(cortes)} cortes): P(pCR) = {probabilidade:.3f}")

        # Três casas decimais: as probabilidades ficam perto do limiar e duas casas pareceriam empate.
        previsto = int(probabilidade >= limiar)
        linhas.append(f"Limiar {limiar:.3f}  ->  classe prevista: pCR {'PRESENTE' if previsto else 'AUSENTE'}")
        real = self.rotulo_real(exame)
        if real is not None:
            linhas.append(f"Classe real na planilha: pCR {'presente' if real else 'ausente'} "
                          f"({'acerto' if real == previsto else 'erro'})")

        visualizador.mostrar_gradcam(mapa, f"Grad-CAM do corte {corte} | P(pCR) = {prob_do_corte:.3f}")
        self.mostrar_resultado("\n".join(linhas))

    # ----- b) aumento de dados e desbalanceamento -----

    def mostrar_aumento_de_dados(self):
        """Abre uma janela com as 10 variações do corte mostrado."""
        visualizador = self.visualizador
        if visualizador.volume is None:
            self.mostrar_resultado("Abra uma imagem antes de gerar o aumento de dados.", "red")
            return

        janela = ctk.CTkToplevel(self.root)
        janela.title(f"Aumento de dados - {visualizador.nome_arquivo} - corte {visualizador.corte_atual}")
        janela.geometry("1100x520")

        figura = Figure(figsize=(11, 4.8))
        eixos = figura.subplots(2, 5)
        variacoes = gerar_aumentos(visualizador.volume[visualizador.corte_atual])
        # VARIACOES alterna original/espelhada; a linha de cima mostra as originais e a de baixo as espelhadas.
        for numero, (descricao, imagem) in enumerate(variacoes):
            eixo = eixos[numero % 2][numero // 2]
            eixo.imshow(imagem, cmap="gray")
            eixo.set_title(descricao, fontsize=9)
            eixo.axis("off")
        figura.tight_layout()

        canvas = FigureCanvasTkAgg(figura, master=janela)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    def mostrar_desbalanceamento(self):
        """Abre uma janela com a contagem das classes em cada partição e o peso usado na loss."""
        try:
            pasta_dataset = PASTA_DATASET
            if self.visualizador.exame["fases"]:
                pasta_dataset = os.path.dirname(os.path.dirname(self.visualizador.exame["fases"][0]))
            contagens = verificar_desbalanceamento(ler_tabela(pasta_dataset))
        except Exception as erro:
            self.mostrar_resultado(f"Não foi possível ler a planilha: {erro}", "red")
            return

        linhas = [f"{'partição':<12}{'sem pCR':>10}{'com pCR':>10}{'% com pCR':>12}"]
        for nome, sem_pcr, com_pcr in contagens:
            linhas.append(f"{nome:<12}{sem_pcr:>10}{com_pcr:>10}{100 * com_pcr / max(sem_pcr + com_pcr, 1):>11.1f}%")
        _, sem_pcr, com_pcr = contagens[0]
        linhas += ["",
                   f"No treino há {sem_pcr / max(com_pcr, 1):.1f} pacientes sem pCR para cada um com pCR.",
                   "Tratamento: a classe pCR entra na função de perda (BCE) com esse peso."]

        janela = ctk.CTkToplevel(self.root)
        janela.title("Desbalanceamento das classes")
        janela.geometry("560x260")
        ctk.CTkLabel(janela, text="\n".join(linhas), font=("Courier", 14), justify="left").pack(padx=20, pady=20)

    # ----- d) treino pela interface -----

    def treinar_rede_selecionada(self):
        """Treina, em segundo plano, a rede escolhida nos menus e mostra o andamento em uma janela."""
        if self.treinando:
            return
        self.treinando = True
        nome_rede, regime, estrategia = self.modelo_selecionado()

        janela = ctk.CTkToplevel(self.root)
        janela.title(f"Treino - {nome_rede} | {regime} | {estrategia}")
        janela.geometry("820x480")
        self.caixa_log = ctk.CTkTextbox(janela, font=("Courier", 12))
        self.caixa_log.pack(fill="both", expand=True, padx=10, pady=10)

        def tarefa():
            with contextlib.redirect_stdout(EscritorDeFila(self.fila_log)):
                try:
                    treinar_redes([nome_rede], [regime], [estrategia])
                except Exception as erro:
                    print(f"\nErro no treino: {erro}")
            self.modelos.pop((nome_rede, regime, estrategia), None)   # força recarregar os pesos novos
            self.treinando = False

        threading.Thread(target=tarefa, daemon=True).start()
        self.root.after(200, self.atualizar_log)

    def atualizar_log(self):
        """Passa para a janela de treino o texto que a thread de treino já produziu."""
        try:
            while True:
                texto = self.fila_log.get_nowait()
                if self.caixa_log is not None and self.caixa_log.winfo_exists():
                    self.caixa_log.insert("end", texto)
                    self.caixa_log.see("end")
        except queue.Empty:
            pass
        if self.treinando or not self.fila_log.empty():
            self.root.after(200, self.atualizar_log)


def abrir_interface():
    root = ctk.CTk()
    InterfaceMRI(root)
    root.mainloop()


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Classificação de pCR em DCE-MRI. Sem opções, abre a interface.")
    parser.add_argument("--treinar", action="store_true", help="treina e avalia as redes (treino final)")
    parser.add_argument("--ajustar", action="store_true", help="roda as tentativas de ajuste de hiperparâmetros")
    parser.add_argument("--redes", nargs="+", default=list(REDES), choices=list(REDES))
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    parser.add_argument("--estrategias", nargs="+", default=ESTRATEGIAS, choices=ESTRATEGIAS)
    parser.add_argument("--taxa-aprendizado", type=float, dest="taxa_aprendizado")
    parser.add_argument("--decaimento-pesos", type=float, dest="decaimento_pesos")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--tamanho-lote", type=int, dest="tamanho_lote")
    parser.add_argument("--semente", type=int)
    argumentos = parser.parse_args()

    # Hiperparâmetros passados na linha de comando substituem os de HIPERPARAMETROS.
    hiper = {chave: getattr(argumentos, chave) for chave in HIPERPARAMETROS if getattr(argumentos, chave) is not None}

    if argumentos.ajustar:
        ajustar_hiperparametros(argumentos.redes, argumentos.regimes)
    elif argumentos.treinar:
        treinar_redes(argumentos.redes, argumentos.regimes, argumentos.estrategias, hiper)
    else:
        abrir_interface()


if __name__ == "__main__":
    main()
