# utils/analysis.py
# Contém a lógica de análise "cara" e as funções de plotagem pós-processamento.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import re

# Velocidade da luz para conversão
c = 299792458.0 

def analyze_peak_properties(S_matrix_total, frequencies):
    """
    Executa uma análise "cara" em um único espectro para encontrar
    o pico real e a largura de banda de -3dB.
    
    Analisa a REFLETIVIDADE (|S11|^2).
    """
    try:
        reflectivity = np.abs(S_matrix_total[0, 0, :])**2
        R_max = np.max(reflectivity)
        
        if R_max < 0.01: 
            return 0.0, 0.0
            
        peak_index = np.argmax(reflectivity)
        f_peak_real_hz = frequencies[peak_index]
        real_peak_wl_nm = (c / f_peak_real_hz) * 1e9
        
        R_3dB_level = R_max * 0.5
        indices_above_3dB = np.where(reflectivity > R_3dB_level)[0]
        
        if indices_above_3dB.size < 2:
            real_bw_hz = 0.0
        else:
            min_idx = np.min(indices_above_3dB) 
            max_idx = np.max(indices_above_3dB) 
            
            f_high = frequencies[min_idx]
            f_low = frequencies[max_idx]
            
            real_bw_hz = abs(f_high - f_low) 
            
        return real_peak_wl_nm, real_bw_hz
        
    except Exception as e:
        print(f"!!! Erro em analyze_peak_properties: {e}")
        return 0.0, 0.0


def run_full_analysis(csv_file_path):
    """
    Carrega dados de um CSV (versão simples), gera um heatmap e um
    pairplot, e salva ambos como arquivos PNG na subpasta do experimento.
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"  [Plotagem] Preparando gráficos de correlação para {len(df)} indivíduos...")
        
        all_possible_dynamic_cols = [
            'Lambda', 'DC', 'w', 'w_c', 'N', 'Fitness', 
            's', 'l', 'height', 'total_length', 'fitness_score' 
        ]
        
        cols_for_correlation = [col for col in all_possible_dynamic_cols if col in df.columns]
        df_analysis = df[cols_for_correlation]

        # --- COMPARTIMENTAÇÃO: Direciona a saída para a subpasta ---
        output_directory = os.path.dirname(csv_file_path)
        base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
        
        # Extrai o timestamp para encontrar a pasta correta
        match = re.search(r'(\d{8}_\d{6})', base_filename)
        if match:
            timestamp_str = match.group(1)
            target_folder = os.path.join(output_directory, f"results_{timestamp_str}")
            # Se por acaso a pasta não existir ainda, salva na raiz do simulation_results
            if not os.path.exists(target_folder):
                target_folder = output_directory
        else:
            target_folder = output_directory

        heatmap_output_path = os.path.join(target_folder, f"{base_filename}_heatmap.png")
        pairplot_output_path = os.path.join(target_folder, f"{base_filename}_pairplot.png")

        # --- 1. Heatmap de Correlação ---
        correlation_matrix = df_analysis.corr()
        plt.figure(figsize=(10, 8)) 
        sns.heatmap(
            correlation_matrix, annot=True, cmap='coolwarm',
            fmt=".2f", linewidths=.5
        )
        plt.title('Matriz de Correlação entre Parâmetros e Fitness')
        plt.savefig(heatmap_output_path, bbox_inches='tight')
        plt.close()
        
        # --- 2. Pairplot ---
        pair_plot = sns.pairplot(df_analysis, diag_kind='kde')
        pair_plot.figure.suptitle('Análise Visual de Pares entre Parâmetros e Fitness', y=1.02)
        pair_plot.savefig(pairplot_output_path)
        plt.close()

    except FileNotFoundError:
        print(f"Erro: O arquivo '{csv_file_path}' não foi encontrado.")
    except KeyError as e:
        print(f"Erro: A coluna {e} não foi encontrada no CSV. Verifique 'all_possible_dynamic_cols'.")
    except Exception as e:
        print(f"Ocorreu um erro durante a análise: {e}")


if __name__ == '__main__':
    file_to_analyze = "caminho/para/seu/full_optimization_data_...csv"
    if os.path.exists(file_to_analyze):
        run_full_analysis(file_to_analyze)
    else:
        print(f"Arquivo de análise de exemplo não encontrado: {file_to_analyze}")