import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt


def gerar_aumentos_nii(fatia_imagem):
    """
    Recebe uma imagem 2D e gera as 10 variações exigidas pelo trabalho.

    Variações:
        - Rotações: -20, -10, 0, 10 e 20 graus.
        - Para cada rotação: imagem original e imagem espelhada.

    Retorna:
        Lista contendo tuplas no formato:
        (tipo, angulo, imagem)
    """

    if fatia_imagem.ndim != 2:
        raise ValueError("A imagem deve ser uma matriz 2D.")

    angulos = [-20, -10, 0, 10, 20]

    variacoes = []

    # Espelhamento horizontal
    fatia_espelhada = np.fliplr(fatia_imagem)

    for angulo in angulos:

        # Imagem original
        rotacao_original = rotate(
            fatia_imagem,
            angle=angulo,
            reshape=False,
            mode='nearest'
        )

        variacoes.append(
            ("Original", angulo, rotacao_original)
        )

        # Imagem espelhada
        rotacao_espelhada = rotate(
            fatia_espelhada,
            angle=angulo,
            reshape=False,
            mode='nearest'
        )

        variacoes.append(
            ("Espelhada", angulo, rotacao_espelhada)
        )

    return variacoes
