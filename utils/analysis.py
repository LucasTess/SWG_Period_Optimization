# utils/analysis.py
# Contém a lógica de análise "cara" e as funções de plotagem pós-processamento.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Velocidade da luz para conversão
c = 299792458.0 

def analyze_peak_properties(S_matrix_total, frequencies):
    """
    Executa uma análise "cara" em um único espectro para encontrar
    o pico real e a largura de banda de -3dB.
    
    Analisa a REFLETIVIDADE (|S11|^2).
    
    Retorna:
        real_peak_wl_nm (float): O comprimento de onda (nm) do pico máximo.
        real_bw_hz (float): A largura de banda de -3dB (em Hz).
    """
    try:
        # 1. Analisa a Refletividade (Potência)
        reflectivity = np.abs(S_matrix_total[0, 0, :])**2
        
        # 2. Encontra o Pico Real
        R_max = np.max(reflectivity)
        
        # Se o pico for insignificante, não é um filtro real.
        if R_max < 0.01: 
            return 0.0, 0.0
            
        peak_index = np.argmax(reflectivity)
        f_peak_real_hz = frequencies[peak_index]
        real_peak_wl_nm = (c / f_peak_real_hz) * 1e9 # Converte para nm
        
        # 3. Encontra a Largura de Banda Real (-3dB)
        R_3dB_level = R_max * 0.5
        
        # Encontra todos os índices onde a refletividade está acima do nível de -3dB
        indices_above_3dB = np.where(reflectivity > R_3dB_level)[0]
        
        if indices_above_3dB.size < 2:
            # O pico é muito fino (1 ponto ou menos)
            real_bw_hz = 0.0
        else:
            # Encontra as bordas deste grupo de índices
            min_idx = np.min(indices_above_3dB) # Índice da frequência mais alta
            max_idx = np.max(indices_above_3dB) # Índice da frequência mais baixa
            
            # Frequências são descendentes, então min_idx é f_high, max_idx é f_low
            f_high = frequencies[min_idx]
            f_low = frequencies[max_idx]
            
            real_bw_hz = abs(f_high - f_low) # Garante valor positivo
            
        return real_peak_wl_nm, real_bw_hz
        
    except Exception as e:
        print(f"!!! Erro em analyze_peak_properties: {e}")
        return 0.0, 0.0


def run_full_analysis(csv_file_path):
    """
    Carrega dados de um CSV (versão simples), gera um heatmap e um
    pairplot, e salva ambos como arquivos PNG.
    
    [CORRIGIDO] Esta versão agora seleciona apenas as colunas dinâmicas
    (parâmetros genéticos + Fitness) para a correlação, ignorando
    parâmetros de configuração estáticos como 'target_center_nm'.
    """
    try:
        df = pd.read_csv(csv_file_path)
        print("Dados carregados com sucesso!")
        print(f"Total de indivíduos analisados: {len(df)}")
        
        # --- [INÍCIO DA CORREÇÃO] ---
        
        # 1. Define a lista de todos os parâmetros genéticos possíveis E o fitness
        #    que você deseja correlacionar.
        #    (Baseado nos seus parâmetros: 'Lambda', 'DC', 'w', 'w_c', 'N', 'Fitness')
        #    (E nos nossos parâmetros anteriores: 's', 'l', 'height', 'total_length')
        all_possible_dynamic_cols = [
            'Lambda', 'DC', 'w', 'w_c', 'N', 'Fitness', # Seus parâmetros atuais
            's', 'l', 'height', 'total_length', 'fitness_score' # Nossos parâmetros anteriores
        ]
        
        # 2. Filtra a lista para incluir apenas as colunas que existem neste CSV
        #    Isso torna o script robusto para diferentes experimentos
        cols_for_correlation = [col for col in all_possible_dynamic_cols if col in df.columns]

        print(f"Colunas selecionadas para análise de correlação: {cols_for_correlation}")

        # 3. O 'df_analysis' agora é criado APENAS com essas colunas dinâmicas.
        #    A lógica que adicionava colunas estáticas foi removida.
        df_analysis = df[cols_for_correlation]

        # --- [FIM DA CORREÇÃO] ---

        # Define os caminhos de saída
        output_directory = os.path.dirname(csv_file_path)
        base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
        heatmap_output_path = os.path.join(output_directory, f"{base_filename}_heatmap.png")
        pairplot_output_path = os.path.join(output_directory, f"{base_filename}_pairplot.png")

        # --- 1. Heatmap de Correlação ---
        print(f"\nGerando Heatmap de Correlação...")
        # Agora 'df_analysis' só contém as colunas corretas
        correlation_matrix = df_analysis.corr()
        plt.figure(figsize=(10, 8)) 
        sns.heatmap(
            correlation_matrix, annot=True, cmap='coolwarm',
            fmt=".2f", linewidths=.5
        )
        plt.title('Matriz de Correlação entre Parâmetros e Fitness')
        plt.savefig(heatmap_output_path, bbox_inches='tight')
        plt.close()
        print(f"-> Heatmap salvo em: {heatmap_output_path}")
        
        # --- 2. Pairplot ---
        print("\nGerando Pairplot... Isso pode levar alguns segundos.")
        # O pairplot agora também usará o DataFrame filtrado
        pair_plot = sns.pairplot(df_analysis, diag_kind='kde')
        pair_plot.figure.suptitle('Análise Visual de Pares entre Parâmetros e Fitness', y=1.02)
        pair_plot.savefig(pairplot_output_path)
        plt.close()
        print(f"-> Pairplot salvo em: {pairplot_output_path}")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{csv_file_path}' não foi encontrado.")
    except KeyError as e:
        print(f"Erro: A coluna {e} não foi encontrada no CSV. Verifique 'all_possible_dynamic_cols'.")
    except Exception as e:
        print(f"Ocorreu um erro durante a análise: {e}")


if __name__ == '__main__':
    # Exemplo de como chamar a função de análise
    file_to_analyze = "caminho/para/seu/full_optimization_data_...csv"
    if os.path.exists(file_to_analyze):
        run_full_analysis(file_to_analyze)
    else:
        print(f"Arquivo de análise de exemplo não encontrado: {file_to_analyze}")