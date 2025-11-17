# main.py (Modularizado para ser chamado por um supervisor)

import sys
import os
import datetime
import shutil
import pandas as pd
import numpy as np
import traceback
import copy # Importa o módulo copy

# <--- MUDANÇA: Caminho para a API do Lumerical (MODE/FDTD usam o mesmo)
_lumapi_module_path = "C:\\Program Files\\Lumerical\\v241\\api\\python"

if _lumapi_module_path not in sys.path:
    sys.path.append(_lumapi_module_path)

import lumapi
# --- Importações dos módulos personalizados ---
from utils.genetic import GeneticOptimizer
from utils.experiment_recorder import record_experiment_results
from utils.lumerical_workflow import simulate_generation_lumerical
from utils.fitness_functions import (
    DeltaAmpStrategy, BandpassStrategy, LowpassStrategy, HighpassStrategy,
    ReflectionBandStrategy
)
from utils.file_handler import clean_simulation_directory
from utils.analysis import run_full_analysis, analyze_peak_properties

# --- [NOVO] Configuração Padrão ---
# Toda a configuração foi movida para este dicionário para que
# possa ser facilmente importada e modificada pelo supervisor.
DEFAULT_CONFIG = {
    "file_paths": {
        "original_lms_file_name": "SWG_period_EME.lms",
        "geometry_lsf_script_name": "create_guide_EME.lsf",
        "simulation_lsf_script_name": "run_simu_guide_EME.lsf",
        "simulation_results_directory_name": "simulation_results"
    },
    "ga_params": {
        "population_size": 4,
        "mutation_rate": 0.2,
        "num_generations": 4,
        "enable_convergence_check": True,
        "convergence_patience_ratio": 0.2, # % de num_generations
        "min_convergence_patience": 20
    },
    "ga_ranges": {
        "Lambda_range": (0.2e-6, 0.4e-6),
        "DC_range": (0.1, 0.9),
        "w_range": (0.4e-6, 0.6e-6),
        "w_c_range_max_ratio": 0.8, # w_c < 0.8 * w
        "N_range": (2, 500)
    },
    "fitness_params": {
        "strategy_name": "reflection_band", # "highpass", "lowpass", "bandpass", "reflection_band"
        "cutoff_wl_nm": 1565,
        "center_wl_nm": 1500,
        "bandwidth_nm": 5,
        "transition_bw_nm": 20,
        "weights": {
            "rejection": 0.50,
            "passband": 0.20,
            "transition": 0.30
        }
    },
    "run_settings": {
        "clean_temp_files": True,
        "lumerical_hide_ui": True
    }
}
# --- Fim da Configuração Padrão ---


# --- [NOVO] Lógica Principal Movida para uma Função ---
def run_optimization(config: dict):
    """
    Executa um experimento de otimização completo com base
    na configuração fornecida.
    """
    
    # --- 1. Desempacotar Configurações ---
    fp = config['file_paths']
    ga_p = config['ga_params']
    ga_r = config['ga_ranges']
    fit_p = config['fitness_params']
    run_s = config['run_settings']

    _project_directory = os.getcwd()
    
    # Diretórios
    _temp_directory = os.path.join(_project_directory, "temp")
    os.makedirs(_temp_directory, exist_ok=True)
    _simulation_results_directory = os.path.join(_project_directory, fp['simulation_results_directory_name'])
    
    # Cria um subdiretório para este experimento específico (baseado no alvo)
    experiment_name = f"{fit_p['strategy_name']}_{fit_p['center_wl_nm']}nm_{fit_p['bandwidth_nm']}nm"
    _exp_output_directory = os.path.join(_simulation_results_directory, experiment_name)
    os.makedirs(_exp_output_directory, exist_ok=True)

    # Caminhos de arquivos
    _original_lms_path = os.path.join(_project_directory, fp['original_lms_file_name'])
    _temp_lms_base_path = os.path.join(_temp_directory, "guide_temp_base.lms")
    _geometry_lsf_script_path = os.path.join(_project_directory, "resources", fp['geometry_lsf_script_name'])
    _simulation_lsf_script_path = os.path.join(_project_directory, "resources", fp['simulation_lsf_script_name'])
    
    # Parâmetros do GA
    population_size = ga_p['population_size']
    mutation_rate = ga_p['mutation_rate']
    num_generations = ga_p['num_generations']
    enable_convergence_check = ga_p['enable_convergence_check']
    CONVERGENCE_PATIENCE = max(
        ga_p['min_convergence_patience'],
        int(num_generations * ga_p['convergence_patience_ratio'])
    )

    # Ranges do GA
    w_c_range = (1e-7, ga_r['w_range'][1] * ga_r['w_c_range_max_ratio'])
    
    # --- 2. Conversão de Unidades e Lógica de Fitness ---
    c = 299792458
    
    # Conversão para Cutoff (High/Low pass)
    f_cutoff_hz = c / (fit_p['cutoff_wl_nm'] * 1e-9)

    # Conversão para Bandpass (Centro e Largura)
    w_center_m = fit_p['center_wl_nm'] * 1e-9
    w_bw_m = fit_p['bandwidth_nm'] * 1e-9
    f_center_hz = c / w_center_m
    f_lower_edge_hz = c / (w_center_m + (w_bw_m / 2))
    f_upper_edge_hz = c / (w_center_m - (w_bw_m / 2))
    bandwidth_hz = f_upper_edge_hz - f_lower_edge_hz

    # Conversão para Largura de Banda de Transição
    w_trans_bw_m = fit_p['transition_bw_nm'] * 1e-9
    f_trans_edge_hz = c / (w_center_m - (w_trans_bw_m / 2))
    transition_bandwidth_hz = abs(f_trans_edge_hz - f_center_hz) * 2
    
    print("--------------------------------------------------------------------------")
    print(f"Iniciando Otimização para: {experiment_name}")
    print(f"Estratégia: {fit_p['strategy_name']}")
    print(f"--> Alvo Central: {f_center_hz/1e12:.2f} THz ({fit_p['center_wl_nm']} nm)")
    print(f"--> Largura de Banda Alvo: {bandwidth_hz/1e12:.4f} THz ({fit_p['bandwidth_nm']} nm)")
    print(f"--> Largura de Transição: {transition_bandwidth_hz/1e12:.4f} THz ({fit_p['transition_bw_nm']} nm)")
    print(f"--> Pesos: Rej={fit_p['weights']['rejection']}, Pass={fit_p['weights']['passband']}, Trans={fit_p['weights']['transition']}")
    print(f"--> GA: {population_size} indivíduos, {num_generations} gerações.")
    print(f"--> Paciência de Convergência: {CONVERGENCE_PATIENCE} gerações")
    print("--------------------------------------------------------------------------")
    
    # Instanciação da Estratégia de Fitness
    fitness_calculator = None
    w = fit_p['weights']
    
    if fit_p['strategy_name'] == "delta_amp":
        fitness_calculator = DeltaAmpStrategy()
    elif fit_p['strategy_name'] == "highpass":
        fitness_calculator = HighpassStrategy(
            f_cutoff=f_cutoff_hz,
            transition_bandwidth=transition_bandwidth_hz,
            w_rejection=w['rejection'], w_passband=w['passband'], w_transition=w['transition']
        )
    elif fit_p['strategy_name'] == "lowpass":
        fitness_calculator = LowpassStrategy(
            f_cutoff=f_cutoff_hz,
            transition_bandwidth=transition_bandwidth_hz,
            w_rejection=w['rejection'], w_passband=w['passband'], w_transition=w['transition']
        )
    elif fit_p['strategy_name'] == "bandpass":
        fitness_calculator = BandpassStrategy(
            f_center=f_center_hz, bandwidth=bandwidth_hz,
            transition_bandwidth=transition_bandwidth_hz,
            w_rejection=w['rejection'], w_passband=w['passband'], w_transition=w['transition']
        )
    elif fit_p['strategy_name'] == "reflection_band":
        fitness_calculator = ReflectionBandStrategy(
            f_center=f_center_hz, bandwidth=bandwidth_hz,
            transition_bandwidth=transition_bandwidth_hz,
            w_rejection=w['rejection'], w_passband=w['passband'], w_transition=w['transition']
        )
    else:
        raise ValueError(f"Estratégia de fitness '{fit_p['strategy_name']}' é inválida.")

    # --- 3. Execução da Otimização ---
    shutil.copy(_original_lms_path, _temp_lms_base_path)
    print(f"Copiado {_original_lms_path} para {_temp_lms_base_path}")

    optimizer = GeneticOptimizer(
        population_size, mutation_rate, num_generations,
        ga_r['Lambda_range'], ga_r['DC_range'], ga_r['w_range'], w_c_range, ga_r['N_range']
    )
    optimizer.initialize_population()
    current_population = optimizer.population

    experiment_start_time = datetime.datetime.now()
    # Salva os resultados no subdiretório específico do experimento
    full_data_csv_path = os.path.join(_exp_output_directory, "full_optimization_data.csv")
    #best_fitness_csv_path = os.path.join(_exp_output_directory, "best_fitness_history.csv")


    generations_processed = 0
    all_individuals_data = []
    best_fitness_so_far = -float('inf')
    generations_without_improvement = 0
    best_fitness_history = [] # Para plotagem

    try:
        with lumapi.MODE(hide=run_s['lumerical_hide_ui']) as mode:
            for gen_num in range(num_generations):
                generations_processed += 1
                print(f"\n--- Processando Geração {gen_num + 1}/{num_generations} ---")
                
                all_S_matrices_for_gen, frequencies = simulate_generation_lumerical(
                    mode, current_population, _temp_lms_base_path,
                    _geometry_lsf_script_path, _simulation_lsf_script_path,
                    _temp_directory
                )
                
                print("\n  [Job Manager] Pós-processando os resultados da geração...")
                
                fitness_scores_for_gen = []
                
                if frequencies is None:
                    print("!!! Erro Crítico: 'frequencies' é None. A simulação pode ter falhado.")
                    fitness_scores_for_gen = [-np.inf] * len(current_population)
                else:
                    for S_matrix in all_S_matrices_for_gen:
                        if S_matrix is None:
                            fitness_scores_for_gen.append(-np.inf)
                            continue
                        try:
                            fitness_score = fitness_calculator.calculate(S_matrix, frequencies)
                        except Exception as e:
                            print(f"!!! Erro no cálculo do fitness para um indivíduo: {e}")
                            fitness_score = -np.inf
                        fitness_scores_for_gen.append(fitness_score)

                # Bloco de Análise do Melhor Indivíduo
                real_peak_wl_nm = 0.0
                real_bw_hz = 0.0
                if frequencies is not None and fitness_scores_for_gen:
                    try:
                        best_gen_index = np.argmax(fitness_scores_for_gen)
                        best_gen_S_matrix = all_S_matrices_for_gen[best_gen_index]
                        if best_gen_S_matrix is not None:
                            real_peak_wl_nm, real_bw_hz = analyze_peak_properties(best_gen_S_matrix, frequencies)
                    except Exception as e:
                        print(f"!!! Erro ao analisar o melhor indivíduo da geração: {e}")

                # --- [CORREÇÃO APLICADA] ---
                
                # 1. Salva uma cópia dos dados da geração ATUAL (antes de evoluir)
                population_before_evolution = copy.deepcopy(current_population)
                scores_for_this_generation = copy.deepcopy(fitness_scores_for_gen)

                # 2. Evoluir (cria a PRÓXIMA geração)
                try:
                    # Passa os scores da geração atual para o método evolve
                    current_population = optimizer.evolve(fitness_scores_for_gen)
                    best_fitness_history.append(optimizer.best_fitness) # Salva o melhor fitness da história
                except ValueError as e:
                    print(f"!!! Erro na evolução da população: {e}")
                    break
                
                print(f"  [Relatório] Salvando relatório e CSV para a Geração {gen_num + 1}...")
                
                # 3. Registra os dados da geração que acabamos de processar
                record_experiment_results(
                    # Caminhos e Tempo
                    output_directory=_simulation_results_directory,
                    full_data_csv_path=full_data_csv_path,
                    experiment_start_time=experiment_start_time,
                    
                    # Estado do Otimizador
                    optimizer_instance=optimizer,
                    generations_processed=generations_processed,
                    
                    # Dados da Geração (para CSV e Análise)
                    all_individuals_data_list=all_individuals_data, # Passa a lista mestre
                    current_population=current_population,
                    fitness_scores_for_gen=fitness_scores_for_gen, 

                    # --- [NOVO] Passa os dados analisados para o JSON ---
                    real_peak_wl_nm=real_peak_wl_nm,
                    real_bw_hz=real_bw_hz,           

                    # Configs de Parâmetros (para JSON)
                    Lambda_range=ga_r['Lambda_range'],
                    DC_range=ga_r['DC_range'],
                    w_range=ga_r['w_range'],
                    w_c_range=w_c_range,
                    N_range=ga_r['N_range'],
                    
                    # Configs de Fitness (para JSON e CSV)
                    fitness_strategy_name=fit_p['strategy_name'],
                    center_wl_nm=fit_p['center_wl_nm'],
                    bandwidth_nm=fit_p['center_bandwidth_nm'],
                    transition_bw_nm=fit_p["transition_bw_nm"],
                    weight_rej=WEIGHT_REJECTION,
                    weight_pass=WEIGHT_PASSBAND,
                    weight_trans=WEIGHT_TRANSITION
                )
                # --- [FIM DA CORREÇÃO] ---

                # Lógica de convergência
                if enable_convergence_check:
                    current_best_fitness = optimizer.best_fitness
                    if current_best_fitness > best_fitness_so_far:
                        print(f"  [Convergência] ✅ Novo melhor fitness encontrado: {current_best_fitness:.4e}. Reiniciando contador.")
                        best_fitness_so_far = current_best_fitness
                        generations_without_improvement = 0
                    else:
                        generations_without_improvement += 1
                        print(f"  [Convergência] ⏳ Nenhuma melhoria no fitness. Gerações sem melhoria: {generations_without_improvement}/{CONVERGENCE_PATIENCE}")

                    if generations_without_improvement >= CONVERGENCE_PATIENCE:
                        print(f"\n  [Convergência] 🛑 O melhor fitness não melhorou por {CONVERGENCE_PATIENCE} gerações consecutivas.")
                        print("  [Convergência] Otimização considerada convergente. Encerrando.")
                        break

        print("\n--- Otimização Concluída ---")
        if optimizer.best_individual:
            print(f"Melhor cromossomo encontrado: {optimizer.best_individual}")
            print(f"Melhor Fitness Score atingido: {optimizer.best_fitness:.4e}")
        else:
            print("Nenhum melhor indivíduo encontrado durante a otimização.")
        
        if run_s['clean_temp_files']:
            clean_simulation_directory(_temp_directory, file_extension=".lms")
            clean_simulation_directory(_temp_directory, file_extension=".log")
            if os.path.exists(_temp_lms_base_path):
                os.remove(_temp_lms_base_path)
                print(f"\n[Limpeza Final] Arquivo base removido: {_temp_lms_base_path}")
        
        # Retorna os resultados para o supervisor
        return optimizer.best_fitness, full_data_csv_path

    except Exception as e:
        print(f"!!! Erro fatal no script principal de otimização: {e}")
        traceback.print_exc()
        return -np.inf, None

# --- [NOVO] Bloco de Execução Principal ---
# Este bloco só é executado quando você roda `python main.py`
if __name__ == "__main__":
    
    print("Executando main.py como script principal com configurações padrão.")
    
    # Chama a função de otimização com a configuração padrão
    run_optimization(DEFAULT_CONFIG)