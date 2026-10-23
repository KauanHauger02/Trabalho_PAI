import nibabel as nib
 

def extrairTodosOsCortes(arquivo,corte):
    return selecionarCorte(arquivo, corte)

def selecionarCorte(arquivo, corte):
    
    imagem = arquivo #recebe a imagem tridimensional (o path)
    img = nib.load(imagem)

    if (corte == "sagital"): #escolha do tipo de corte
        return Sagital(img)
    elif(corte == "coronal"):
        return Coronal(img)
    elif(corte == "axial"):
        return Axial(img)

#as funções abaixo retornam um array contendo os dados de cada corte em uma certa dimensão

def Sagital (img):
    
    data = img.get_fdata()
    cortes = [] #vetor que vai guardar os metadados de cada camada daquele sentido

    for i in range(data.shape[0]): #Pega quantos cortes sagitais existem e vai iterando sobre eles
        corte = data[i,:,:] # pega os metadados do corte sagital I 
        cortes.append(corte) # adiciona esses metadados no array contendo todos os cortes

    return cortes

def Coronal (img):
    
    data = img.get_fdata()
    cortes = []

    for i in range(data.shape[1]):
        corte = data[:,i,:]
        cortes.append(corte)

    return cortes

def Axial (img):
    
    data = img.get_fdata()
    cortes = []

    for i in range (data.shape[2]):
        corte = data[:,:,i]
        print(corte.shape)
        cortes.append(corte)

    return cortes