# main.py
import sys
import os
import datetime
import shutil
import pandas as pd
import numpy as np
import traceback
import copy 
import re

# Embora o main.py já não abra o modo diretamente, 
# mantemos o path para garantir que as dependências resolvem bem.
_lumapi_module_path = "C:\\Program Files\\Lumerical\\v241\\api\\python"
if _lumapi_module_path not in sys.path:
    sys.path.append(_lumapi_module_path)

from utils.genetic import GeneticOptimizer
from utils.woa import WhaleOptimizer
from utils.experiment_recorder import record_experiment_results
from utils.lumerical_workflow import simulate_generation_lumerical
from utils.fitness_functions import ReflectionBandStrategy
from utils.file_handler import clean_simulation_directory
from utils.analysis import run_full_analysis, analyze_peak_properties

def run_optimization(config: dict):
    fp = config['file_paths']
    ga_p = config['ga_params']
    ga_r = config['ga_ranges']
    fit_p = config['fitness_params']
    run_s = config['run_settings']
    
    # --- CHECKPOINT RESCUE ---
    checkpoint = config.get('checkpoint', None)
    resume_gen = 0
    if checkpoint:
        resume_gen = checkpoint['start_generation']
        rescued_pop = checkpoint['population']
        original_csv_path = checkpoint['original_csv_path']
        print(f"[*] Modo Resgate Ativado! Retomando da Geração {resume_gen}.")

    _project_directory = os.getcwd()
    _temp_directory = os.path.join(_project_directory, "temp")
    os.makedirs(_temp_directory, exist_ok=True)
    _simulation_results_directory = os.path.join(_project_directory, fp['simulation_results_directory_name'])
    os.makedirs(_simulation_results_directory, exist_ok=True)

    _original_lms_path = os.path.join(_project_directory, fp['original_lms_file_name'])
    _temp_lms_base_path = os.path.join(_temp_directory, "guide_temp_base.lms")
    _geometry_lsf_script_path = os.path.join(_project_directory, "resources", fp['geometry_lsf_script_name'])
    _simulation_lsf_script_path = os.path.join(_project_directory, "resources", fp['simulation_lsf_script_name'])
    
    population_size = ga_p['population_size']
    mutation_rate = ga_p['mutation_rate']
    num_generations = ga_p['num_generations']
    enable_convergence_check = ga_p['enable_convergence_check']
    CONVERGENCE_PATIENCE = max(
        ga_p['min_convergence_patience'],
        int(num_generations * ga_p['convergence_patience_ratio'])
    )

    w_range = ga_r['w_range']
    w_c_range = (1e-7, w_range[1] * ga_r['w_c_range_max_ratio'])
    
    c = 299792458
    FITNESS_STRATEGY_NAME = fit_p['strategy_name']
    CENTER_WAVELENGTH_NM = fit_p['center_wl_nm']
    BANDWIDTH_NM = fit_p['bandwidth_nm']
    TRANSITION_BANDWIDTH_NM = fit_p['transition_bw_nm']
    
    w = fit_p['weights']
    WEIGHT_REJECTION = w['rejection']
    WEIGHT_PASSBAND = w['passband']
    WEIGHT_TRANSITION = w['transition']

    w_center_m = CENTER_WAVELENGTH_NM * 1e-9
    w_bw_m = BANDWIDTH_NM * 1e-9
    f_center_hz = c / w_center_m
    f_lower_edge_hz = c / (w_center_m + (w_bw_m / 2))
    f_upper_edge_hz = c / (w_center_m - (w_bw_m / 2))
    bandwidth_hz = f_upper_edge_hz - f_lower_edge_hz

    w_trans_bw_m = TRANSITION_BANDWIDTH_NM * 1e-9
    f_trans_edge_hz = c / (w_center_m - (w_trans_bw_m / 2))
    transition_bandwidth_hz = abs(f_trans_edge_hz - f_center_hz) * 2
    
    # --- Identifica o Otimizador Desejado ---
    opt_type = config.get('optimizer_type', 'GA')

    # Gera o nome do experimento (Nomenclatura Limpa)
    experiment_start_time = datetime.datetime.now()
    if checkpoint:
        full_data_csv_path = original_csv_path
        experiment_prefix = os.path.basename(original_csv_path).replace('_full_data.csv', '')
    else:
        timestamp_str = experiment_start_time.strftime('%Y%m%d_%H%M%S')
        experiment_prefix = f"{CENTER_WAVELENGTH_NM}nm_{BANDWIDTH_NM}nm_{timestamp_str}"
        full_data_csv_path = os.path.join(_simulation_results_directory, f"{experiment_prefix}_full_data.csv")
    
    print("--------------------------------------------------------------------------")
    print(f"A Iniciar Otimização: {experiment_prefix} [{opt_type}]")
    print(f"--> Alvo Central: {CENTER_WAVELENGTH_NM} nm")
    print("--------------------------------------------------------------------------")
    
    if FITNESS_STRATEGY_NAME == "reflection_band":
        fitness_calculator = ReflectionBandStrategy(
            f_center=f_center_hz, bandwidth=bandwidth_hz, 
            transition_bandwidth=transition_bandwidth_hz, 
            w_rejection=WEIGHT_REJECTION, w_passband=WEIGHT_PASSBAND, 
            w_transition=WEIGHT_TRANSITION
        )
    else:
        raise ValueError("Estratégia inválida.")

    shutil.copy(_original_lms_path, _temp_lms_base_path)

    # --- INSTANCIAÇÃO DO OTIMIZADOR (FACTORY) ---
    if opt_type == 'WOA':
        optimizer = WhaleOptimizer(
            population_size, mutation_rate, num_generations,
            ga_r['Lambda_range'], ga_r['DC_range'], ga_r['w_range'], w_c_range, ga_r['N_range']
        )
        if checkpoint:
            optimizer.current_generation = resume_gen
    else:
        optimizer = GeneticOptimizer(
            population_size, mutation_rate, num_generations,
            ga_r['Lambda_range'], ga_r['DC_range'], ga_r['w_range'], w_c_range, ga_r['N_range']
        )
    
    # --- INJEÇÃO DA POPULAÇÃO ---
    if checkpoint:
        optimizer.population = rescued_pop
    else:
        optimizer.initialize_population()
        
    current_population = optimizer.population

    generations_processed = resume_gen
    all_individuals_data = []
    
    if checkpoint:
        try:
            df_old = pd.read_csv(original_csv_path)
            all_individuals_data = df_old.to_dict('records')
        except Exception:
            pass
            
    best_fitness_so_far = -float('inf')
    generations_without_improvement = 0

    try:
        # --- LOOP ADAPTATIVO ---
        for gen_num in range(resume_gen, num_generations):
            generations_processed += 1
            print(f"\n--- A Processar Geração {gen_num + 1}/{num_generations} ---")
            
            # [MODIFICADO] A chamada agora é feita sem o parâmetro "mode" 
            # e com o parâmetro hide_ui no final
            # Chama a função nativa hibrida sem scripts LSF externos
            all_S_matrices_for_gen, frequencies = simulate_generation_lumerical(
                current_population, w_center_m, w_bw_m, _temp_lms_base_path,
                _temp_directory, run_s['lumerical_hide_ui']
            )
            
            fitness_scores_for_gen = []
            if frequencies is None:
                fitness_scores_for_gen = [-np.inf] * len(current_population)
            else:

                for S_matrix in all_S_matrices_for_gen:
                    if S_matrix is None:
                        fitness_scores_for_gen.append(-np.inf)
                        continue
                    
                    try:
                        fitness_score = fitness_calculator.calculate(S_matrix, frequencies)
                    except Exception as e:
                        print(f"!!! Erro no fitness: {e}")
                        fitness_score = -np.inf
                    fitness_scores_for_gen.append(fitness_score)


            real_peak_wl_nm = 0.0
            real_bw_hz = 0.0
            if frequencies is not None and fitness_scores_for_gen:
                try:
                    best_gen_index = np.argmax(fitness_scores_for_gen)
                    best_gen_S_matrix = all_S_matrices_for_gen[best_gen_index]
                    if best_gen_S_matrix is not None:
                        real_peak_wl_nm, real_bw_hz = analyze_peak_properties(best_gen_S_matrix, frequencies)
                except Exception as e:
                    pass
            
            population_before_evolution = copy.deepcopy(current_population)
            scores_for_this_generation = copy.deepcopy(fitness_scores_for_gen)

            try:
                # Proteção contra falha total na população
                if all(s == -np.inf for s in fitness_scores_for_gen):
                    print("⚠️ Aviso: Nenhuma simulação válida nesta geração. Pulando evolução.")
                    continue # Pula para a próxima geração sem evoluir baleias corrompidas
                current_population = optimizer.evolve(scores_for_this_generation)
            except ValueError as e:
                break
            
            record_experiment_results(
                output_directory=_simulation_results_directory,
                full_data_csv_path=full_data_csv_path,
                experiment_start_time=experiment_start_time,
                optimizer_instance=optimizer,
                generations_processed=generations_processed,
                all_individuals_data_list=all_individuals_data,
                current_population=population_before_evolution,
                fitness_scores_for_gen=scores_for_this_generation,
                real_peak_wl_nm=real_peak_wl_nm,
                real_bw_hz=real_bw_hz,            
                Lambda_range=ga_r['Lambda_range'],
                DC_range=ga_r['DC_range'],
                w_range=ga_r['w_range'],
                w_c_range=w_c_range,
                N_range=ga_r['N_range'],
                fitness_strategy_name=FITNESS_STRATEGY_NAME,
                center_wl_nm=CENTER_WAVELENGTH_NM,
                bandwidth_nm=BANDWIDTH_NM,
                transition_bw_nm=TRANSITION_BANDWIDTH_NM,
                weight_rej=WEIGHT_REJECTION,
                weight_pass=WEIGHT_PASSBAND,
                weight_trans=WEIGHT_TRANSITION,
                optimizer_type=opt_type
            )
            
            if all_individuals_data:
                run_full_analysis(full_data_csv_path) 

            if enable_convergence_check:
                current_best_fitness = optimizer.best_fitness
                if current_best_fitness > best_fitness_so_far:
                    best_fitness_so_far = current_best_fitness
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1

                if generations_without_improvement >= CONVERGENCE_PATIENCE:
                    print(f"\n  [Convergência] 🛑 Otimização considerada convergente.")
                    break

        print("\n--- Otimização Concluída ---")
        if run_s['clean_temp_files']:
            clean_simulation_directory(_temp_directory, file_extension=".lms")
            clean_simulation_directory(_temp_directory, file_extension=".log")
            if os.path.exists(_temp_lms_base_path):
                try:
                    os.remove(_temp_lms_base_path)
                except:
                    pass
        
        return optimizer.best_fitness, full_data_csv_path

    except Exception as e:
        traceback.print_exc()
        return -np.inf, None

if __name__ == "__main__":
    pass