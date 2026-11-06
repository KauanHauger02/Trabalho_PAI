# Declaração de uso de ferramentas de IA Generativa

**Disciplina:** Processamento e Análise de Imagens (PUC Minas, ICEI, Prof. Alexei Machado)
**Trabalho:** Classificação da resposta ao tratamento do câncer de mama usando redes neurais
**Grupo:** Gnn (PREENCHER)

| Nome completo | Matrícula | Curso | Campus |
|---|---|---|---|
| Bernardo PREENCHER | PREENCHER | Ciência da Computação | PREENCHER |
| Diego PREENCHER | PREENCHER | Ciência da Computação | PREENCHER |
| Kauan PREENCHER | PREENCHER | Ciência da Computação | PREENCHER |
| Túlio PREENCHER | PREENCHER | Ciência da Computação | PREENCHER |

Esta declaração lista todo uso de IA generativa no trabalho, em qualquer fase, com a ferramenta,
a finalidade e os textos enviados (prompts), como pede o item (c) das considerações gerais do enunciado.

> **Atenção, grupo:** só a seção 1 está preenchida, porque é o uso que ficou registrado.
> Cada integrante precisa preencher a sua parte na seção 2 antes da entrega, inclusive para dizer que não usou.

---

## 1. Uso feito por Túlio

**Ferramenta:** Claude Code (Anthropic), modelo Claude Fable 5.1, dentro do VS Code.
**Datas:** 04/11/2026 e 05/11/2026.

### 1.1 Prompt de 04/11/2026: revisão do repositório

> eu ja tenho o github do projeto https://github.com/KauanHauger02/Trabalho_PAI.git, os meninos ja comecaram
> com suas branchs pessoais, cada um fez uma parte, da uma olhada e o pdf do enunciado dizendo o que tem que
> fazer esta na pasta local aqui
>
> Voce consegue acessar o git atraves do meu github, em outros projetos ja uso assim

**Finalidade:** comparar o que já estava nas branches do grupo com o enunciado.
**Resultado:** uma lista do que cada integrante tinha feito e do que faltava: máscara ao lado da imagem,
classificação com Grad-CAM na interface, plano de exibição, arquivo único, ajuste de hiperparâmetros,
relatório, esta declaração e repositório privado.

### 1.2 Prompt de 05/11/2026: correção dos pontos levantados

> Acho que ja criaram uma branch e um integrante falou que ia atulizar alguma parte disso que voce apontou
> mas pode arrumar esses topicos, ai vou mandar la novamente no grupo oque fizemos

**Finalidade:** corrigir os pontos da lista acima.
**O que a ferramenta produziu nesta etapa:**

| Parte do trabalho | O que a IA fez |
|---|---|
| Código (`trabalho_pai.py`) | Juntou em um arquivo único os módulos escritos pelo grupo: leitura, redes, treino, métricas e Grad-CAM (Bernardo), extração de cortes e interface (Diego), aumento de dados, análise de desbalanceamento e visualizador (Kauan). |
| Código, partes novas | Escreveu: exibição da máscara ao lado da imagem acompanhando o corte z; classificação com Grad-CAM na interface; janelas de aumento de dados, de desbalanceamento e de treino; escolha do limiar de decisão na validação; cálculo da AUC; medida de sobreposição entre Grad-CAM e tumor; rotina de ajuste de hiperparâmetros; gravação e leitura dos modelos treinados. |
| Análise de dados | Verificou o formato do dataset, encontrou a inconsistência do paciente ISPY1_1139 na planilha, executou as tentativas de ajuste de hiperparâmetros e o treino final e montou as tabelas de resultados. |
| Redação | Escreveu a primeira versão do relatório em LaTeX (pasta `relatorio/`), do `README.md` e a atualização do `ENTENDIMENTO.md`. |
| Declaração | Escreveu a primeira versão deste arquivo. |

---

## 2. Uso feito pelos demais integrantes (PREENCHER)

Para cada uso, informar ferramenta, data, finalidade e os prompts. Se não houve uso, escrever "Não utilizei".

### Bernardo
PREENCHER (inclui dizer se o `neuralNetwork.py` e o `ENTENDIMENTO.md` tiveram apoio de IA).

### Diego
PREENCHER

### Kauan
PREENCHER

---

## 3. Revisão pelo grupo

PREENCHER: descrever como o grupo revisou, testou e alterou o que foi produzido com apoio de IA.
