# utils/experiment_recorder.py
# Centraliza o salvamento de JSON (com análise) e CSV (rápido).

import os
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Importa a função de análise que será chamada 1x por geração
from utils.analysis import analyze_peak_properties 

# Velocidade da luz (para conversão nm <-> Hz)
c = 299792458.0 

def record_experiment_results(
    # --- Parâmetros de Configuração e Caminho ---
    output_directory,
    full_data_csv_path,
    experiment_start_time,
    
    # --- Estado do Otimizador ---
    optimizer_instance,
    generations_processed,
    
    # --- Dados da Geração Atual (para CSV) ---
    all_individuals_data_list,  # A lista mestre de dados
    current_population,
    fitness_scores_for_gen,
    
    # --- Dados para Análise do Melhor Indivíduo ---
    real_peak_wl_nm,
    real_bw_hz,
    
    # --- Configs de Parâmetros (para JSON) ---
    Lambda_range, DC_range, w_range, w_c_range, N_range,
    
    # --- Configs de Fitness (para JSON e CSV) ---
    fitness_strategy_name,
    center_wl_nm,
    bandwidth_nm,
    transition_bw_nm,
    weight_rej,
    weight_pass,
    weight_trans, 
    **kwargs
):
    """
    Registra os resultados do experimento (JSON) e atualiza o log de dados
    completo (CSV) para a geração atual.
    """
    
    # --- Gera nomes de arquivos (para JSON e Gráfico) ---
    timestamp_str = experiment_start_time.strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_directory, f"experiment_results_{timestamp_str}.json")
    plot_path = os.path.join(output_directory, f"fitness_history_{timestamp_str}.png")

    current_time = datetime.datetime.now()
    duration = current_time - experiment_start_time

    # --- Análise do Melhor Indivíduo (para JSON) ---
    best_gen_analysis = {}
    try:
        # Calcula a largura de banda real em nm (para facilitar a leitura no JSON)
        real_bw_nm = 0.0
        if real_bw_hz > 0 and real_peak_wl_nm > 0:
            f_peak_hz = c / (real_peak_wl_nm * 1e-9)
            f_low = f_peak_hz - (real_bw_hz / 2)
            f_high = f_peak_hz + (real_bw_hz / 2)
            wl_low_nm = (c / f_low) * 1e9
            wl_high_nm = (c / f_high) * 1e9
            real_bw_nm = abs(wl_low_nm - wl_high_nm)

        best_gen_analysis = {
            "real_peak_wl_nm": real_peak_wl_nm,
            "real_bw_hz": real_bw_hz,
            "real_bw_nm": real_bw_nm
        }
    except Exception as e:
        print(f"!!! Erro ao calcular real_bw_nm para JSON: {e}")

    # --- 1. Lógica do JSON ---
    results_data = {
        "experiment_start_time": experiment_start_time.isoformat(),
        "last_update": current_time.isoformat(),
        "current_duration": str(duration),
        "generations_processed": generations_processed,
        "population_size": optimizer_instance.population_size,
        "mutation_rate": optimizer_instance.mutation_rate,
        "max_generations_set": optimizer_instance.generations,
        "best_individual_so_far": optimizer_instance.best_individual,
        "best_fitness_so_far": optimizer_instance.best_fitness,
        "analysis_of_best_in_gen": best_gen_analysis,  # Dados extras aqui
        "parameter_ranges": {
            "Lambda": Lambda_range, "DC": DC_range, "w": w_range,
            "w_c": w_c_range, "N": N_range
        },
        "fitness_configuration": {
            "strategy": fitness_strategy_name,
            "center_wavelength_nm": center_wl_nm,
            "bandwidth_nm": bandwidth_nm,
            "transition_bandwidth_nm": transition_bw_nm,
            "weight_rejection": weight_rej,
            "weight_passband": weight_pass,
            "weight_transition": weight_trans
        },
        "fitness_history": optimizer_instance.fitness_history
    }

    # Salva o arquivo JSON
    try:
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=4)
    except Exception as e:
        print(f"!!! Erro ao salvar/atualizar resultados do experimento (JSON): {e}")

    
    # --- 2. Lógica do CSV (RÁPIDA - Sem análise "cara") ---
    
    # Adiciona os dados da geração atual à lista mestre
    for i, chromosome in enumerate(current_population):
        individual_data = chromosome.copy()
        individual_data['Fitness'] = fitness_scores_for_gen[i]
        individual_data['generation'] = generations_processed
        individual_data['fitness_strategy'] = fitness_strategy_name
        
        # Parâmetros de fitness (para rastreamento)
        individual_data['target_center_nm'] = center_wl_nm
        individual_data['target_bw_nm'] = bandwidth_nm
        individual_data['target_trans_bw_nm'] = transition_bw_nm
        individual_data['w_rej'] = weight_rej
        individual_data['w_pass'] = weight_pass
        individual_data['w_trans'] = weight_trans
        
        # A análise "cara" foi removida daqui
        
        all_individuals_data_list.append(individual_data)

    # Salva o CSV completo
    if all_individuals_data_list:
        try:
            df_all_data = pd.DataFrame(all_individuals_data_list)
            df_all_data.to_csv(full_data_csv_path, index=False)
            print(f"  [Análise] Dados de {len(all_individuals_data_list)} indivíduos atualizados em CSV.")
        except Exception as e:
            print(f"!!! Erro ao salvar/atualizar log de dados (CSV): {e}")


    # --- 3. Lógica do Gráfico de Fitness (Inalterada) ---
    if optimizer_instance.fitness_history:
        plt.figure(figsize=(10, 6))
        generations = range(1, len(optimizer_instance.fitness_history) + 1)
        plt.plot(generations, optimizer_instance.fitness_history, marker='o', linestyle='-')
        plt.title(f'Histórico de Fitness (Atualizado em: {current_time.strftime("%H:%M:%S")})')
        plt.xlabel('Geração')
        plt.ylabel('Melhor Fitness')
        plt.grid(True)
        try:
            plt.savefig(plot_path)
        except Exception as e:
            print(f"!!! Erro ao salvar/atualizar gráfico de fitness: {e}")
        finally:
            plt.close()  # Libera memória
    else:
        print("Nenhum histórico de fitness para plotar.")