#este arquivo serve meramente para testar código que será executado em outros arquivos e não deve
#pertencer ao trabalho final, caso queiram testar uma biblioteca ou algo do tipo rodem aqui

import nibabel as nib 
import os 
import numpy as np
import math

path = os.path.join("C:/Users/Diego/Desktop/trabalho PAI/Base de dados/BreastDCEDL_ISPY1_min_crop/dce/ISPY1_1004_spy1_vis1_dce_aqc_0.nii")


img = nib.load(path)


class extrairCorteCentral:
    
    def __int__(self,arquivo,corte):
        
        self.selecionarCorte(arquivo, corte)

    def selecionarCorte(self, arquivo, corte):
     
        imagem = arquivo #recebe o arquivo com as imagens tridimensionais
        img = nib.load(imagem)

        if (corte == "sagital"):
            return self.Sagital(img)
        elif(corte == "coronal"):
            return self.Coronal(img)
        else:
            return self.Axial(img)
     
    def Sagital(self, img):
        #os eixos dos arquivos das imagens estão ordenados em nos eixos R A S => R = SAGITAL, A = CORONAL, S = AXIAL
        
        data = img.get_fdata() # data recebe um array em que os valores sao ordenados em [R, A, S]
        eixo = data.shape[0] # pega o eixo sagital
        eixo = eixo / 2 # pega a camada do meio do eixo sagital
        eixo = math.trunc(eixo) #trunca o numero para baixo em caso de numero impar 
        return eixo

    def Coronal(self, img):
   
        data = img.get_fdata() 
        eixo = data.shape[1]
        eixo = eixo / 2 
        eixo = math.trunc(eixo) 
        return eixo

    def Axial(self, img):
   
        data = img.get_fdata() 
        eixo = data.shape[2] 
        eixo = eixo / 2 
        eixo = math.trunc(eixo) 
        return eixo

    
x = extrairCorteCentral()

print(x.selecionarCorte(path, "axial"))