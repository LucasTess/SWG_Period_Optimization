# main.py (Motor de Otimização - Versão Híbrida com Seleção de Solver e Workflow)

import sys
import os
import datetime
import shutil
import pandas as pd
import numpy as np
import traceback
import copy 

# --- Caminho para a API do Lumerical ---
_lumapi_module_path = "C:\\Program Files\\Lumerical\\v241\\api\\python"
if _lumapi_module_path not in sys.path:
    sys.path.append(_lumapi_module_path)

import lumapi

# --- Importações dos módulos personalizados ---
from utils.genetic import GeneticOptimizer
from utils.experiment_recorder import record_experiment_results
from utils.fitness_functions import ReflectionBandStrategy
from utils.file_handler import clean_simulation_directory
from utils.analysis import run_full_analysis

# Importação dos dois workflows distintos com apelidos
from utils.lumerical_workflow_EME import simulate_generation_lumerical as simulate_EME
from utils.lumerical_workflow_FDTD import simulate_generation_lumerical as simulate_FDTD

def run_optimization(config: dict, stop_check=None):
    """
    Executa a otimização selecionando dinamicamente entre EME (Uniforme) 
    e FDTD (Apodizado).
    """
    fp = config['file_paths']
    ga_p = config['ga_params']
    ga_r = config['ga_ranges'] 
    fit_p = config['fitness_params']
    run_s = config['run_settings']

    _project_directory = os.getcwd()
    _temp_directory = os.path.join(_project_directory, "temp")
    os.makedirs(_temp_directory, exist_ok=True)
    _simulation_results_directory = os.path.join(_project_directory, fp['simulation_results_directory_name'])
    os.makedirs(_simulation_results_directory, exist_ok=True)

    # --- Seleção Dinâmica do Arquivo Base e Solver ---
    if ga_r['mode'] == "apodized":
        file_name = "bragg_guide_FDTD.fsp"
        solver_type = "FDTD"
        simulate_func = simulate_FDTD
    else:
        file_name = "bragg_guide_EME.lms"
        solver_type = "EME"
        simulate_func = simulate_EME

    _original_file_path = os.path.join(_project_directory, file_name)
    _extension = os.path.splitext(file_name)[1]
    _temp_base_path = os.path.join(_temp_directory, f"guide_temp_base{_extension}")
    
    _geometry_lsf_script_path = os.path.join(_project_directory, "resources", fp['geometry_lsf_script_name'])
    _simulation_lsf_script_path = os.path.join(_project_directory, "resources", fp['simulation_lsf_script_name'])
    
    # --- Lógica de Fitness ---
    c = 299792458
    f_center_hz = c / (fit_p['center_wl_nm'] * 1e-9)
    bw_nm = fit_p['bandwidth_nm']
    f_lower = c / ((fit_p['center_wl_nm'] + bw_nm/2) * 1e-9)
    f_upper = c / ((fit_p['center_wl_nm'] - bw_nm/2) * 1e-9)
    bandwidth_hz = abs(f_upper - f_lower)
    
    trans_bw_nm = fit_p['transition_bw_nm']
    f_trans = c / ((fit_p['center_wl_nm'] - trans_bw_nm/2) * 1e-9)
    transition_bandwidth_hz = abs(f_trans - f_center_hz) * 2
    
    experiment_start_time = datetime.datetime.now()
    timestamp_str = experiment_start_time.strftime('%Y%m%d_%H%M%S')
    experiment_prefix = f"{ga_r['mode']}_{fit_p['strategy_name']}_{fit_p['center_wl_nm']}nm_{timestamp_str}"

    print(f"\n--------------------------------------------------------------------------")
    print(f"Iniciando Otimização: {ga_r['mode'].upper()}")
    print(f"Solver: {solver_type} | Arquivo Base: {file_name}")
    print(f"GA: {ga_p['population_size']} indivíduos, {ga_p['num_generations']} gerações.")
    print(f"--------------------------------------------------------------------------")
    
    fitness_calculator = ReflectionBandStrategy(
        f_center=f_center_hz, bandwidth=bandwidth_hz,
        transition_bandwidth=transition_bandwidth_hz,
        w_rejection=fit_p['weights']['rejection'], 
        w_passband=fit_p['weights']['passband'], 
        w_transition=fit_p['weights']['transition']
    )

    # --- Inicialização do AG ---
    if not os.path.exists(_original_file_path):
        raise FileNotFoundError(f"Arquivo base não encontrado: {_original_file_path}")
        
    shutil.copy(_original_file_path, _temp_base_path)
    
    optimizer = GeneticOptimizer(ga_p['population_size'], ga_p['mutation_rate'], ga_p['num_generations'], ga_r)
    optimizer.initialize_population()
    current_population = optimizer.population

    full_data_csv_path = os.path.join(_simulation_results_directory, f"{experiment_prefix}_full_data.csv")
    generations_processed = 0
    all_individuals_data = [] 
    mode_session = None 

    try:
        # Abre a sessão correta
        if solver_type == "FDTD":
            mode_session = lumapi.FDTD(hide=run_s['lumerical_hide_ui'])
        else:
            mode_session = lumapi.MODE(hide=run_s['lumerical_hide_ui'])
        
        for gen_num in range(ga_p['num_generations']):
            if stop_check and stop_check():
                print("\n--- Parada solicitada. Encerrando motor... ---")
                break 

            generations_processed += 1
            print(f"\n--- Geração {generations_processed}/{ga_p['num_generations']} ---")
            
            # Chama o workflow correspondente (EME Serial ou FDTD Parallel)
            all_S_matrices, frequencies = simulate_func(
                mode_session, current_population, _temp_base_path,
                _geometry_lsf_script_path, _simulation_lsf_script_path,
                _temp_directory, mode_type=ga_r['mode'], stop_check=stop_check
            )
            
            # Cálculo de Fitness
            fitness_scores = []
            if frequencies is None:
                fitness_scores = [-np.inf] * len(current_population)
            else:
                for S in all_S_matrices:
                    fitness_scores.append(fitness_calculator.calculate(S, frequencies) if S is not None else -np.inf)

            # Evolução e Registro
            pop_before = copy.deepcopy(current_population)
            scores_before = copy.deepcopy(fitness_scores)
            current_population = optimizer.evolve(scores_before)

            # --- [FIX] Filtramos ga_r para passar apenas o que o recorder espera ---
            # O recorder espera apenas os nomes das chaves que terminam com '_range'
            clean_ranges = {k: v for k, v in ga_r.items() if k.endswith('_range')}
            
            record_experiment_results(
                output_directory=_simulation_results_directory,
                full_data_csv_path=full_data_csv_path,
                experiment_start_time=experiment_start_time,
                optimizer_instance=optimizer,
                generations_processed=generations_processed,
                all_individuals_data_list=all_individuals_data,
                current_population=pop_before,
                fitness_scores_for_gen=scores_before,
                real_peak_wl_nm=0.0, real_bw_hz=0.0,
                **clean_ranges, # Passa apenas os ranges filtrados
                fitness_strategy_name=fit_p['strategy_name'],
                center_wl_nm=fit_p['center_wl_nm'], bandwidth_nm=fit_p['bandwidth_nm'],
                transition_bw_nm=fit_p['transition_bw_nm'],
                weight_rej=fit_p['weights']['rejection'], weight_pass=fit_p['weights']['passband'], weight_trans=fit_p['weights']['transition']
            )

        return optimizer.best_fitness, full_data_csv_path

    except Exception as e:
        print(f"!!! Erro fatal no motor de otimização: {e}")
        traceback.print_exc()
        return -np.inf, None

    finally:
        if mode_session:
            try:
                mode_session.close()
            except:
                pass