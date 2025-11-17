# main.py (Modularizado para ser um "motor" puro)
# [CORRIGIDO] Alinhado com a assinatura de função exata
# do seu 'utils/experiment_recorder.py'

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

# --- Lógica Principal Movida para uma Função ---
# Este script agora é um módulo e só define esta função "trabalhadora".
# Ele não tem mais um DEFAULT_CONFIG global.
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
    os.makedirs(_simulation_results_directory, exist_ok=True) # Garante que a pasta exista

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
    w_range = ga_r['w_range'] # Pega o w_range para calcular o w_c_range
    w_c_range = (1e-7, w_range[1] * ga_r['w_c_range_max_ratio'])
    
    # --- 2. Conversão de Unidades e Lógica de Fitness ---
    c = 299792458
    
    # Desempacota os parâmetros de fitness
    FITNESS_STRATEGY_NAME = fit_p['strategy_name']
    CUTOFF_WAVELENGTH_NM = fit_p['cutoff_wl_nm']
    CENTER_WAVELENGTH_NM = fit_p['center_wl_nm']
    BANDWIDTH_NM = fit_p['bandwidth_nm']
    TRANSITION_BANDWIDTH_NM = fit_p['transition_bw_nm']
    
    # Desempacota os pesos
    w = fit_p['weights']
    WEIGHT_REJECTION = w['rejection']
    WEIGHT_PASSBAND = w['passband']
    WEIGHT_TRANSITION = w['transition']

    # Conversão para Cutoff (High/Low pass)
    f_cutoff_hz = c / (CUTOFF_WAVELENGTH_NM * 1e-9)

    # Conversão para Bandpass (Centro e Largura)
    w_center_m = CENTER_WAVELENGTH_NM * 1e-9
    w_bw_m = BANDWIDTH_NM * 1e-9
    f_center_hz = c / w_center_m
    f_lower_edge_hz = c / (w_center_m + (w_bw_m / 2))
    f_upper_edge_hz = c / (w_center_m - (w_bw_m / 2))
    bandwidth_hz = f_upper_edge_hz - f_lower_edge_hz

    # Conversão para Largura de Banda de Transição
    w_trans_bw_m = TRANSITION_BANDWIDTH_NM * 1e-9
    f_trans_edge_hz = c / (w_center_m - (w_trans_bw_m / 2))
    transition_bandwidth_hz = abs(f_trans_edge_hz - f_center_hz) * 2
    
    # Gera o timestamp para este experimento
    experiment_start_time = datetime.datetime.now()
    timestamp_str = experiment_start_time.strftime('%Y%m%d_%H%M%S')
    
    # Cria o prefixo único para este experimento
    experiment_prefix = f"{fit_p['strategy_name']}_{fit_p['center_wl_nm']}nm_{fit_p['bandwidth_nm']}nm_{timestamp_str}"
    
    print("--------------------------------------------------------------------------")
    print(f"Iniciando Otimização: {experiment_prefix}")
    print(f"Estratégia: {FITNESS_STRATEGY_NAME}")
    print(f"--> Alvo Central: {f_center_hz/1e12:.2f} THz ({CENTER_WAVELENGTH_NM} nm)")
    print(f"--> Largura de Banda Alvo: {bandwidth_hz/1e12:.4f} THz ({BANDWIDTH_NM} nm)")
    print(f"--> Largura de Transição: {transition_bandwidth_hz/1e12:.4f} THz ({TRANSITION_BANDWIDTH_NM} nm)")
    print(f"--> Pesos: Rej={WEIGHT_REJECTION}, Pass={WEIGHT_PASSBAND}, Trans={WEIGHT_TRANSITION}")
    print(f"--> GA: {population_size} indivíduos, {num_generations} gerações.")
    print(f"--> Paciência de Convergência: {CONVERGENCE_PATIENCE} gerações")
    print("--------------------------------------------------------------------------")
    
    # Instanciação da Estratégia de Fitness
    fitness_calculator = None
    
    if FITNESS_STRATEGY_NAME == "delta_amp":
        fitness_calculator = DeltaAmpStrategy()
    elif FITNESS_STRATEGY_NAME == "highpass":
        fitness_calculator = HighpassStrategy(
            f_cutoff=f_cutoff_hz,
            transition_bandwidth=transition_bandwidth_hz,
            w_rejection=WEIGHT_REJECTION, w_passband=WEIGHT_PASSBAND, w_transition=WEIGHT_TRANSITION
        )
    elif FITNESS_STRATEGY_NAME == "lowpass":
        fitness_calculator = LowpassStrategy(
            f_cutoff=f_cutoff_hz,
            transition_bandwidth=transition_bandwidth_hz,
            w_rejection=WEIGHT_REJECTION, w_passband=WEIGHT_PASSBAND, w_transition=WEIGHT_TRANSITION
        )
    elif FITNESS_STRATEGY_NAME == "bandpass":
        fitness_calculator = BandpassStrategy(
            f_center=f_center_hz, bandwidth=bandwidth_hz,
            transition_bandwidth=transition_bandwidth_hz,
            w_rejection=WEIGHT_REJECTION, w_passband=WEIGHT_PASSBAND, w_transition=WEIGHT_TRANSITION
        )
    elif FITNESS_STRATEGY_NAME == "reflection_band":
        fitness_calculator = ReflectionBandStrategy(
            f_center=f_center_hz, bandwidth=bandwidth_hz,
            transition_bandwidth=transition_bandwidth_hz,
            w_rejection=WEIGHT_REJECTION, w_passband=WEIGHT_PASSBAND, w_transition=WEIGHT_TRANSITION
        )
    else:
        raise ValueError(f"Estratégia de fitness '{FITNESS_STRATEGY_NAME}' é inválida.")

    # --- 3. Execução da Otimização ---
    shutil.copy(_original_lms_path, _temp_lms_base_path)
    print(f"Copiado {_original_lms_path} para {_temp_lms_base_path}")

    optimizer = GeneticOptimizer(
        population_size, mutation_rate, num_generations,
        ga_r['Lambda_range'], ga_r['DC_range'], ga_r['w_range'], w_c_range, ga_r['N_range']
    )
    optimizer.initialize_population()
    current_population = optimizer.population

    # Define o caminho do CSV de dados completos (o único caminho de que o recorder precisa)
    full_data_csv_path = os.path.join(_simulation_results_directory, f"{experiment_prefix}_full_data.csv")
    
    generations_processed = 0
    all_individuals_data = [] # A lista mestre que é passada para o recorder
    best_fitness_so_far = -float('inf')
    generations_without_improvement = 0
    # best_fitness_history não é mais necessário aqui, pois o optimizer_instance cuida disso

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
                
                # Salva uma cópia dos dados da geração ATUAL (antes de evoluir)
                population_before_evolution = copy.deepcopy(current_population)
                scores_for_this_generation = copy.deepcopy(fitness_scores_for_gen)

                # Evolui para a PRÓXIMA geração
                try:
                    current_population = optimizer.evolve(scores_for_this_generation)
                except ValueError as e:
                    print(f"!!! Erro na evolução da população: {e}")
                    break
                
                print(f"  [Relatório] Salvando relatório e CSV para a Geração {gen_num + 1}...")
                
                # --- [CHAMADA DE FUNÇÃO CORRIGIDA] ---
                # Corresponde exatamente à assinatura da função que você forneceu
                record_experiment_results(
                    # Caminhos e Tempo
                    output_directory=_simulation_results_directory, # Pasta raiz de resultados
                    full_data_csv_path=full_data_csv_path, # Caminho único com timestamp
                    experiment_start_time=experiment_start_time,
                    
                    # Estado do Otimizador
                    optimizer_instance=optimizer,
                    generations_processed=generations_processed,
                    
                    # Dados da Geração (para CSV e Análise)
                    all_individuals_data_list=all_individuals_data, # Passa a lista mestre
                    current_population=population_before_evolution, # Geração N
                    fitness_scores_for_gen=scores_for_this_generation, # Scores da Geração N

                    # Dados analisados para o JSON
                    real_peak_wl_nm=real_peak_wl_nm,
                    real_bw_hz=real_bw_hz,            

                    # Configs de Parâmetros (para JSON)
                    Lambda_range=ga_r['Lambda_range'],
                    DC_range=ga_r['DC_range'],
                    w_range=ga_r['w_range'],
                    w_c_range=w_c_range,
                    N_range=ga_r['N_range'],
                    
                    # Configs de Fitness (para JSON e CSV)
                    fitness_strategy_name=FITNESS_STRATEGY_NAME,
                    center_wl_nm=CENTER_WAVELENGTH_NM,
                    bandwidth_nm=BANDWIDTH_NM,
                    transition_bw_nm=TRANSITION_BANDWIDTH_NM,
                    weight_rej=WEIGHT_REJECTION,
                    weight_pass=WEIGHT_PASSBAND,
                    weight_trans=WEIGHT_TRANSITION
                )
                # --- [FIM DA CORREÇÃO] ---
                
                # Reativa a chamada para análise (pairplot/heatmap)
                if all_individuals_data:
                    run_full_analysis(full_data_csv_path) 
                    print(f"  [Análise] Gráficos de análise atualizados e salvos.")

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
    
    # --- [NOVO] Configuração Padrão ---
    # Definida localmente, apenas para testes. Não é visível para o supervisor.
    DEFAULT_CONFIG = {
        "file_paths": {
            "original_lms_file_name": "SWG_period_EME.lms",
            "geometry_lsf_script_name": "create_guide_EME.lsf",
            "simulation_lsf_script_name": "run_simu_guide_EME.lsf",
            "simulation_results_directory_name": "simulation_results"
        },
        "ga_params": {
            "population_size": 4, # Tamanho pequeno para teste rápido
            "mutation_rate": 0.2,
            "num_generations": 2, # Poucas gerações para teste rápido
            "enable_convergence_check": True,
            "convergence_patience_ratio": 0.2,
            "min_convergence_patience": 20
        },
        "ga_ranges": {
            "Lambda_range": (0.2e-6, 0.4e-6),
            "DC_range": (0.1, 0.9),
            "w_range": (0.4e-6, 0.6e-6),
            "w_c_range_max_ratio": 0.8,
            "N_range": (2, 500)
        },
        "fitness_params": {
            "strategy_name": "reflection_band",
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
    
    # Chama a função de otimização com a configuração padrão
    run_optimization(DEFAULT_CONFIG)