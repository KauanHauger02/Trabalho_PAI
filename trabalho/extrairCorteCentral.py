import nibabel as nib
import numpy as np
import math

def extrairCorteCentral(arquivo, corte):
    return selecionarCorte(arquivo, corte)

def selecionarCorte(arquivo, corte):
    
    imagem = arquivo #recebe a imagem tridimensional
    img = nib.load(imagem)

    if (corte == "sagital"):
        return Sagital(img)
    elif(corte == "coronal"):
        return Coronal(img)
    elif(corte == "axial"):
        return Axial(img)
    
def Sagital(img):
    #os eixos dos arquivos das imagens estão ordenados em nos eixos R A S => R = SAGITAL, A = CORONAL, S = AXIAL
    
    data = img.get_fdata() # data recebe um array em que os valores sao ordenados em [R, A, S]
    eixo = data.shape[0] # pega o eixo sagital
    eixo = eixo / 2 # pega a camada do meio do eixo sagital
    eixo = math.trunc(eixo) #trunca o numero para baixo em caso de numero impar 
    return data[eixo,:,:] #retorna a matriz 2D de dados naquele corte do eixo especificado

def Coronal(img):

    data = img.get_fdata() 
    eixo = data.shape[1]
    eixo = eixo / 2 
    eixo = math.trunc(eixo) 
    return data [:,eixo,:]

def Axial(img):

    data = img.get_fdata() 
    eixo = data.shape[2] 
    eixo = eixo / 2 
    eixo = math.trunc(eixo) 
    return data [:,:,eixo]

