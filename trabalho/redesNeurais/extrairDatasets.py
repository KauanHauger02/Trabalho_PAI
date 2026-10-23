#script para pegar os dados dos arquivos de imagem e transforma los em datasets para serem utilizados pela rede neural convolucional

import os 
import numpy as np
import nibabel as nib
import sys
sys.path.append("../")
from extrairCorteCentral import extrairCorteCentral
from extrairTodosOsCortes import extrairTodosOsCortes

dir_path = os.path.join("C:/Users/Diego/Desktop/trabalho PAI/Base de dados/BreastDCEDL_ISPY1_min_crop/dce") #seleciona o arquivo com as imagens
files_in_dir = os.listdir(dir_path) #itera pelo arquivo pegando as imagens 
num_images = len(files_in_dir) # pega o numero de imagens no arquivo

#esta função retorna o corte central sagital de todas as imagens do dataset
def gerar_dataset_cortes_centrais():
    
    cortes_centrais_sagitais = np.zeros((num_images,256,256), dtype=np.float64) #array que matrizes que armazena os cortes centrais sagitais

    for i, file in enumerate(files_in_dir):
        full_path = os.path.join(dir_path, file)
        cortes_centrais_sagitais[i] = extrairCorteCentral(full_path, "sagital")

    return cortes_centrais_sagitais

#esta função retorna todos os cortes sagitais de todas as imagens
def gerar_dataset_todos_cortes():
    
    todos_os_cortes_todas_imagens = np.zeros((num_images)) #cada posição neste array é um array de matrizes, o array de matrizes contem todos os cortes daquela imagem
    #TODO verificar se é preciso especificar o tipo de dados neste vetor "dtype float 64" pois cada posição é um vetor e não uma matriz de numeros 64 bits
    
    for i, file in enumerate(files_in_dir):
        full_path = os.path.join(dir_path,file)
        cortes_de_uma_imagem = extrairTodosOsCortes(full_path, "sagital")
        todos_os_cortes_todas_imagens[i] = cortes_de_uma_imagem

    return todos_os_cortes_todas_imagens



