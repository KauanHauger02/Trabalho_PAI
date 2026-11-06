# Trabalho_PAI

Trabalho prático de Processamento e Análise de Imagens (PUC Minas): classificação da resposta
patológica completa (pCR) ao tratamento do câncer de mama a partir de DCE-MRI, com redes neurais.

Todo o código está em um arquivo só, como pede o enunciado: [`trabalho_pai.py`](trabalho_pai.py).

## Preparação

1. Python 3.12 e as bibliotecas:
   ```
   python -m venv .venv
   .venv/bin/pip install torch torchvision nibabel scipy numpy pandas scikit-learn matplotlib customtkinter
   ```
   No Windows, use `.venv\Scripts\pip` e `.venv\Scripts\python`.
2. Dataset: baixar `BreastDCEDL_ISPY1_min_crop.tar.gz` (1,3 GB) do
   [Zenodo](https://zenodo.org/records/18114231) e descompactar em `Base de dados/`, de modo que existam
   `Base de dados/BreastDCEDL_ISPY1_min_crop/dce/` e `.../mask/`. A planilha
   `BreastDCEDL_SPY1_min_crop_metadata.csv` já está na raiz do repositório.
3. A pasta `Base de dados/`, o `.venv/` e a pasta `pesos/` não vão para o git.

## Como usar

| Comando | O que faz |
|---|---|
| `python trabalho_pai.py` | Abre a interface gráfica |
| `python trabalho_pai.py --treinar` | Treina e avalia as 3 redes nos 2 regimes e nas 2 estratégias (12 modelos) |
| `python trabalho_pai.py --treinar --redes resnet50 --regimes 3fases --estrategias central` | Treina só uma combinação |
| `python trabalho_pai.py --ajustar` | Roda as tentativas de ajuste de hiperparâmetros (3 sementes cada) |

Na interface:

- **Selecionar pasta MRI:** escolha a pasta do dataset ou a pasta `dce/`. Ao clicar em um arquivo, a imagem
  aparece no plano xy com a segmentação do tumor ao lado, e o controle deslizante percorre os cortes em z.
- **Classificar + Grad-CAM:** classifica o corte mostrado com a rede, o regime e a estratégia escolhidos nos
  menus, e mostra a probabilidade de pCR, a classe prevista, a classe real e o mapa do Grad-CAM.
  Precisa dos pesos em `pesos/`, criados pelo treino.
- **Aumento de dados:** mostra as 10 variações do corte (5 rotações, original e espelhada).
- **Desbalanceamento das classes:** mostra a contagem de cada classe por partição e o peso usado na loss.
- **Treinar rede selecionada:** treina a combinação escolhida nos menus e mostra o andamento.

## Arquivos

| Caminho | Conteúdo |
|---|---|
| `trabalho_pai.py` | Código completo, dividido em 9 etapas |
| `ENTENDIMENTO.md` | Explicação de cada função e de cada escolha, para estudo do grupo |
| `relatorio/` | Relatório em LaTeX no padrão da SBC e o PDF |
| `resultados/metricas.csv` | Métricas do treino final no conjunto de teste, com o limiar 0,5 e com o limiar da validação |
| `resultados/ajuste.csv` e `ajuste_resumo.csv` | Todas as tentativas do ajuste de hiperparâmetros (126 treinos) |
| `resultados/ajuste_convergencia.png` | Curvas de loss das 7 tentativas do ajuste |
| `resultados/convergencia_*.png`, `loss_*.png` | Curvas de convergência do treino final, em grade e por modelo |
| `resultados/gradcam_*.png` | Exemplos de Grad-CAM: um acerto e um erro de cada tipo (VP, VN, FP, FN) por modelo |
| `resultados/log_*.txt` | Saída completa do ajuste e do treino final |
| `resultados/rodada1/` | Resultados da primeira rodada, de 30/10, antes do ajuste |
| `declaracao_uso_iag.md` | Declaração de uso de IA generativa |

## Resultado em uma frase

Os 12 modelos ficaram perto do acaso (AUC de teste entre 0,33 e 0,67). O melhor foi a ResNet50 com
3 fases no corte central (acurácia 0,71). Os detalhes e a discussão estão no relatório e no `ENTENDIMENTO.md`.

## O que não está no repositório

- `Base de dados/`: o dataset.
- `pesos/`: os 12 modelos treinados (614 MB). Para recriar: `python trabalho_pai.py --treinar`, cerca de
  12 minutos em um Mac M4 Pro. Sem essa pasta, o botão "Classificar + Grad-CAM" avisa que falta treinar.

## Problema conhecido

No macOS com o Python do python.org, o download dos pesos do ImageNet pode falhar com
`CERTIFICATE_VERIFY_FAILED`. Resolva executando uma vez o `Install Certificates.command` que vem na pasta
do Python, ou baixando os dois arquivos à mão para `~/.cache/torch/hub/checkpoints/`:

```
curl -L -o ~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth https://download.pytorch.org/models/resnet50-11ad3fa6.pth
curl -L -o ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth https://download.pytorch.org/models/vgg16-397923af.pth
```
