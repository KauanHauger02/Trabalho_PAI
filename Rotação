import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

def gerar_aumentos_nii(fatia_imagem):
    """
    Recebe uma matriz 2D e retorna uma lista com as 10 variações exigidas:
    - Rotações: -20, -10, 0, 10, 20 graus.
    - Versões: Original e Espelhada (flip horizontal).
    """
    angulos = [-20, -10, 0, 10, 20] # Limites e intervalos definidos[cite: 3]
    variacoes = []
    
    # Cria a versão com espelhamento horizontal[cite: 3]
    fatia_espelhada = np.fliplr(fatia_imagem) 
    
    for angulo in angulos:
        # reshape=False mantém as dimensões de entrada (ex: 256x256)[cite: 2]
        # mode='nearest' previne a criação de bordas pretas irreais durante a rotação
        
        # 1. Rotação da imagem original
        rot_orig = rotate(fatia_imagem, angle=angulo, reshape=False, mode='nearest')
        variacoes.append(('Original', angulo, rot_orig))
        
        # 2. Rotação da imagem espelhada
        rot_esp = rotate(fatia_espelhada, angle=angulo, reshape=False, mode='nearest')
        variacoes.append(('Espelhada', angulo, rot_esp))
        
    return variacoes

# ==========================================
# SCRIPT DE TESTE ISOLADO
# ==========================================
if __name__ == "__main__":
    # Criação de uma matriz 256x256 simulando um corte do dataset I-SPY1 MinCrop[cite: 2]
    imagem_teste = np.zeros((256, 256))
    
    # Simulando uma região tumoral descentralizada para perceber o efeito da rotação
    imagem_teste[80:120, 150:180] = 1 
    
    resultados = gerar_aumentos_nii(imagem_teste)
    
    # Plot para validação visual das 10 saídas[cite: 3]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for idx, (tipo, ang, img) in enumerate(resultados):
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f"{tipo}\n{ang}°")
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.show()
