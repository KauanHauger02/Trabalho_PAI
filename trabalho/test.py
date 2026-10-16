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


    
x = extrairCorteCentral()

#print(x.selecionarCorte(path, "axial"))

y = extrairTodosOsCortes()
#print(y.selecionarCorte(path, "axial"))


resultado = y.selecionarCorte(path, "sagital")

print("Quantidade de cortes:", len(resultado))
print("Formato do primeiro corte:", resultado[0].shape)
print("Dados do primeiro corte:", resultado[0])
print("Dados do segundo corte:", resultado[2])
