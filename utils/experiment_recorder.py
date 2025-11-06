# experiment_end.py (Modificado para centralizar JSON e CSV)

import os
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # <--- Importação necessária

def record_experiment_results(
    # --- Parâmetros de Configuração e Caminho ---
    output_directory,
    full_data_csv_path,
    experiment_start_time,
    
    # --- Estado do Otimizador ---
    optimizer_instance,
    generations_processed,
    
    # --- Dados da Geração Atual (para CSV) ---
    all_individuals_data_list, # A lista mestre de dados
    current_population,
    fitness_scores_for_gen,
    
    # --- Configs de Parâmetros (para JSON) ---
    Lambda_range, DC_range, w_range, w_c_range, N_range,
    
    # --- Configs de Fitness (para JSON e CSV) ---
    fitness_strategy_name,
    center_wl_nm,
    bandwidth_nm,
    transition_bw_nm,
    weight_rej,
    weight_pass,
    weight_trans
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

    # --- 1. Lógica do JSON (Atualizada com Goal 1) ---
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
        "parameter_ranges": {
            "Lambda": Lambda_range,
            "DC": DC_range,
            "w": w_range,
            "w_c": w_c_range,
            "N": N_range
        },
        # --- [NOVO] Adiciona a configuração de fitness ao JSON ---
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

    
    # --- 2. Lógica do CSV (Movida do main.py para cá - Goal 2) ---
    
    # Adiciona os dados da geração atual à lista mestre
    for i, chromosome in enumerate(current_population):
        individual_data = chromosome.copy()
        individual_data['Fitness'] = fitness_scores_for_gen[i]
        individual_data['generation'] = generations_processed # (gen_num + 1)
        individual_data['fitness_strategy'] = fitness_strategy_name
        
        # --- [NOVO] Adiciona os parâmetros de fitness ao CSV ---
        # Útil se você mudar os pesos no meio de um experimento
        individual_data['target_center_nm'] = center_wl_nm
        individual_data['target_bw_nm'] = bandwidth_nm
        individual_data['target_trans_bw_nm'] = transition_bw_nm
        individual_data['w_rej'] = weight_rej
        individual_data['w_pass'] = weight_pass
        individual_data['w_trans'] = weight_trans
        
        all_individuals_data_list.append(individual_data)

    # Salva o CSV completo (sobrescrevendo o anterior com os novos dados)
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
            plt.close() # Libera memória
    else:
        print("Nenhum histórico de fitness para plotar.")