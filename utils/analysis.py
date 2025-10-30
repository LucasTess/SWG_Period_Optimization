# analysis.py (Modificado para os novos parâmetros)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_full_analysis(csv_file_path):
    """
    Carrega dados de um CSV, gera um heatmap de correlação e um
    pairplot, e salva ambos como arquivos PNG no mesmo diretório do CSV.
    """
    try:
        # --- PREPARAÇÃO DOS DADOS E NOMES DE ARQUIVO ---
        
        df = pd.read_csv(csv_file_path)
        print("Dados carregados com sucesso!")
        print(f"Total de indivíduos analisados: {len(df)}")
        
        # --- MUDANÇA AQUI ---
        # Atualiza a lista de parâmetros para a análise
        params_and_fitness = ['Lambda', 'DC', 'w', 'w_c', 'N', 'Fitness']
        df_analysis = df[params_and_fitness]

        # Define os caminhos de saída baseados no nome do arquivo de entrada
        output_directory = os.path.dirname(csv_file_path)
        base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
        
        heatmap_output_path = os.path.join(output_directory, f"{base_filename}_heatmap.png")
        pairplot_output_path = os.path.join(output_directory, f"{base_filename}_pairplot.png")

        # --- 1. Heatmap de Correlação (O Resumo) ---
        print(f"\nGerando Heatmap de Correlação...")
        correlation_matrix = df_analysis.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix, 
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=.5
        )
        plt.title('Matriz de Correlação entre Parâmetros e Fitness')
        
        plt.savefig(heatmap_output_path)
        plt.close()
        print(f"-> Heatmap salvo em: {heatmap_output_path}")
        
        # --- 2. Pairplot (A Análise Completa) ---
        print("\nGerando Pairplot... Isso pode levar alguns segundos.")
        
        pair_plot = sns.pairplot(
            df_analysis,
            diag_kind='kde' # Mostra uma curva de densidade na diagonal
        )
        
        pair_plot.figure.suptitle('Análise Visual de Pares entre Parâmetros e Fitness', y=1.02)
        
        pair_plot.savefig(pairplot_output_path)
        plt.close()
        print(f"-> Pairplot salvo em: {pairplot_output_path}")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{csv_file_path}' não foi encontrado.")
    except KeyError as e:
        print(f"Erro: A coluna {e} não foi encontrada no CSV. Verifique 'params_and_fitness'.")
    except Exception as e:
        print(f"Ocorreu um erro durante a análise: {e}")


if __name__ == '__main__':
    # ATUALIZE AQUI com o caminho para o seu arquivo CSV que deseja analisar
    file_to_analyze = "Caminho\\Para\\Seu\\arquivo.csv"
    
    run_full_analysis(file_to_analyze)