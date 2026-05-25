# utils/experiment_recorder.py
import os
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from utils.analysis import analyze_peak_properties 
c = 299792458.0 

def record_experiment_results(
    output_directory, full_data_csv_path, experiment_start_time,
    optimizer_instance, generations_processed, all_individuals_data_list, 
    current_population, fitness_scores_for_gen, real_peak_wl_nm, real_bw_hz,
    Lambda_range, DC_range, w_range, w_c_range, N_range,
    fitness_strategy_name, center_wl_nm, bandwidth_nm, transition_bw_nm,
    weight_rej, weight_pass, weight_trans,
    optimizer_type # <-- NOVO PARÂMETRO
):
    
    csv_filename = os.path.basename(full_data_csv_path)
    match = re.search(r'(\d{8}_\d{6})', csv_filename)
    
    if match:
        timestamp_str = match.group(1)
    else:
        timestamp_str = experiment_start_time.strftime('%Y%m%d_%H%M%S')
        
    experiment_subfolder = os.path.join(output_directory, f"results_{timestamp_str}")
    os.makedirs(experiment_subfolder, exist_ok=True)
        
    results_path = os.path.join(experiment_subfolder, f"experiment_results_{timestamp_str}.json")
    plot_path = os.path.join(experiment_subfolder, f"fitness_history_{timestamp_str}.png")

    current_time = datetime.datetime.now()
    duration = current_time - experiment_start_time

    # --- 1. Atualiza a Lista Mestre para o CSV ---
    for i, chromosome in enumerate(current_population):
        individual_data = chromosome.copy()
        individual_data['Fitness'] = fitness_scores_for_gen[i]
        individual_data['generation'] = generations_processed
        individual_data['optimizer_type'] = optimizer_type  # <-- GRAVA NO CSV
        individual_data['fitness_strategy'] = fitness_strategy_name
        individual_data['target_center_nm'] = center_wl_nm
        individual_data['target_bw_nm'] = bandwidth_nm
        individual_data['target_trans_bw_nm'] = transition_bw_nm
        individual_data['w_rej'] = weight_rej
        individual_data['w_pass'] = weight_pass
        individual_data['w_trans'] = weight_trans
        
        all_individuals_data_list.append(individual_data)

    # --- 2. Sincronização do Histórico ---
    if all_individuals_data_list:
        df_temp = pd.DataFrame(all_individuals_data_list)
        full_history = df_temp.groupby('generation')['Fitness'].max().tolist()
        optimizer_instance.fitness_history = full_history
        
        best_idx = df_temp['Fitness'].idxmax()
        best_row = df_temp.loc[best_idx]
        
        optimizer_instance.best_fitness = float(best_row['Fitness'])
        optimizer_instance.best_individual = {
            'Lambda': float(best_row['Lambda']),
            'DC': float(best_row['DC']),
            'w': float(best_row['w']),
            'w_c': float(best_row['w_c']),
            'N': int(best_row['N'])
        }

    # --- 3. Salva o CSV Completo ---
    if all_individuals_data_list:
        try:
            df_all_data = pd.DataFrame(all_individuals_data_list)
            df_all_data.to_csv(full_data_csv_path, index=False)
            print(f"  [Análise] Dados de {len(all_individuals_data_list)} indivíduos atualizados no CSV bruto.")
        except Exception as e:
            print(f"!!! Erro ao salvar log de dados (CSV): {e}")

    # --- 4. Análise do Melhor da Geração ---
    best_gen_analysis = {}
    try:
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
        pass

    # --- 5. Lógica do JSON ---
    results_data = {
        "experiment_start_time": experiment_start_time.isoformat(),
        "last_update": current_time.isoformat(),
        "optimizer_type": optimizer_type,  # <-- GRAVA NO JSON
        "current_duration": str(duration),
        "generations_processed": generations_processed,
        "population_size": optimizer_instance.population_size,
        "max_generations_set": optimizer_instance.generations,
        "best_individual_so_far": optimizer_instance.best_individual,
        "best_fitness_so_far": optimizer_instance.best_fitness,
        "analysis_of_best_in_gen": best_gen_analysis, 
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

    try:
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=4)
    except Exception as e:
        print(f"!!! Erro ao salvar JSON: {e}")

    # --- 6. Lógica do Gráfico de Fitness ---
    if optimizer_instance.fitness_history:
        plt.figure(figsize=(10, 6))
        generations = range(1, len(optimizer_instance.fitness_history) + 1)
        plt.plot(generations, optimizer_instance.fitness_history, marker='o', linestyle='-')
        plt.title(f'Histórico de Fitness ({optimizer_type})') # Colocando a sigla no título também!
        plt.xlabel('Geração')
        plt.ylabel('Melhor Fitness')
        plt.grid(True)
        try:
            plt.savefig(plot_path)
        except Exception as e:
            pass
        finally:
            plt.close()