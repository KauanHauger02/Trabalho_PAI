#este arquivo serve meramente para testar código que será executado em outros arquivos e não deve
#pertencer ao trabalho final, caso queiram testar uma biblioteca ou algo do tipo rodem aqui

import nibabel as nib 
import os 
import numpy as np
import math
from extrairCorteCentral import extrairCorteCentral
from extrairTodosOsCortes import extrairTodosOsCortes

path = os.path.join("C:/Users/Diego/Desktop/trabalho PAI/Base de dados/BreastDCEDL_ISPY1_min_crop/dce/ISPY1_1001_spy1_vis1_dce_aqc_0.nii")

img = nib.load(path)
print(img.shape)
print(img.ndim)

print(extrairCorteCentral(path, "sagital"))

fatia = extrairCorteCentral(path, "sagital")

print(fatia.shape)
print(fatia.ndim)

# dir_path = os.path.join("C:/Users/Diego/Desktop/trabalho PAI/Base de dados/BreastDCEDL_ISPY1_min_crop/dce") #seleciona o arquivo com as imagens
# files_in_dir = os.listdir(dir_path) #itera pelo arquivo pegando as imagens 

# for file in files_in_dir:
#     full_path = os.path.join(dir_path, file)

#     img = nib.load(full_path)
#     print(img.shape)
#     print(img.ndim)
#     print(img.get_data_dtype())