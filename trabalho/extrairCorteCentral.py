import nibabel as nib
import numpy as np
import math

class extrairCorteCentral:
    
    def __int__(self,corte):
        
        self.selecionarCorte(corte)

    def selecionarCorte(self, arquivo, corte):
     
        imagem = arquivo #recebe a imagem tridimensional
        img = nib.load(imagem)

        if (corte == "sagital"):
            return self.Sagital(img)
        elif(corte == "coronal"):
            return self.Coronal(img)
        elif(corte == "axial"):
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

    
