import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# verifica se existe um acelerador disponível para uso na rede neural, caso negativo utiliza a CPU da máquina 
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" 
print(f"Utilizando {device} no treinamento")

