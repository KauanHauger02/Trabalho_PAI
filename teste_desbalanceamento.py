import pandas as pd

def verificar_desbalanceamento_csv(caminho_csv):
    """
    Lê o arquivo CSV do dataset e contabiliza a distribuição das classes.
    Retorna a contagem da Classe 0 (pCR ausente) e Classe 1 (pCR presente).
    """
    try:
        # 1. Lê a planilha CSV usando o argumento passado para a função
        df = pd.read_csv(caminho_csv)
        
        # A coluna chama-se pCR no .csv
        coluna_classe = 'pCR' 
        
        # 2. Conta quantos pacientes pertencem a cada classe
        contagem = df[coluna_classe].value_counts()
        
        classe_0 = contagem.get(0, 0) # pCR ausente
        classe_1 = contagem.get(1, 0) # pCR presente[cite: 2]
        total_pacientes = classe_0 + classe_1
        
        # 3. Exibe os resultados com percentagens
        print("=== ANÁLISE DE DESBALANCEAMENTO ===")
        print(f"Total de amostras: {total_pacientes}")
        print(f"Classe 0 (pCR Ausente): {classe_0} amostras ({classe_0/total_pacientes*100:.2f}%)")
        print(f"Classe 1 (pCR Presente): {classe_1} amostras ({classe_1/total_pacientes*100:.2f}%)")
        
        if abs(classe_0 - classe_1) / total_pacientes > 0.2: # Diferença maior que 20%
            print("\nConclusão: Existe um desbalanceamento relevante.")
            print("Estratégia recomendada: Calcular pesos diferentes para a função de perda.")
        else:
            print("\nConclusão: O dataset está relativamente balanceado.")
            
        return classe_0, classe_1

    except FileNotFoundError:
        print(f"Erro: O ficheiro {caminho_csv} não foi encontrado.")
        return None, None
    except KeyError:
        print(f"Erro: A coluna '{coluna_classe}' não existe no CSV. Verifica o nome correto.")
        return None, None

# ==========================================
# SCRIPT DE TESTE
# ==========================================
if __name__ == "__main__":
    # COMO O CSV ESTÁ NA MESMA PASTA DO .PY, BASTA COLOCAR O NOME DO ARQUIVO AQUI
    caminho_do_ficheiro = "BreastDCEDL_SPY1_min_crop_metadata.csv" 
    
    # Executa a função
    qtd_classe_0, qtd_classe_1 = verificar_desbalanceamento_csv(caminho_do_ficheiro)