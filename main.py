# main.py (Atualizado para EME e novo fluxo de trabalho)

import sys
import os
import datetime
import shutil
import pandas as pd
import numpy as np

# <--- MUDANÇA: Caminho para a API do Lumerical (MODE/FDTD usam o mesmo)
_lumapi_module_path = "C:\\Program Files\\Lumerical\\v241\\api\\python"

if _lumapi_module_path not in sys.path:
    sys.path.append(_lumapi_module_path)

import lumapi
# --- Importações dos módulos personalizados ---
from utils.genetic import GeneticOptimizer
from utils.experiment_end import record_experiment_results
from utils.lumerical_workflow import simulate_generation_lumerical
from utils.fitness_functions import DeltaAmpStrategy, BandpassStrategy, LowpassStrategy, HighpassStrategy
from utils.file_handler import clean_simulation_directory
from utils.analysis import run_full_analysis

# --- Configurações Globais ---
_project_directory = os.getcwd()
# <--- MUDANÇA: Arquivos base do Lumerical MODE
_original_lms_file_name = "SWG_period_EME.lms" # Use um arquivo .lms base
_geometry_lsf_script_name = "create_guide_EME.lsf"
_simulation_lsf_script_name = "run_simu_guide_EME.lsf"
_simulation_results_directory_name = "simulation_results"

# --- Diretórios ---
_temp_directory = os.path.join(_project_directory, "temp")
os.makedirs(_temp_directory, exist_ok=True)
_simulation_results_directory = os.path.join(_project_directory, _simulation_results_directory_name)
os.makedirs(_simulation_results_directory, exist_ok=True)

# <--- MUDANÇA: Caminhos de arquivos .lms
_original_lms_path = os.path.join(_project_directory, _original_lms_file_name)
_temp_lms_base_path = os.path.join(_temp_directory, "guide_temp_base.lms")
_geometry_lsf_script_path = os.path.join(_project_directory, "resources", _geometry_lsf_script_name)
_simulation_lsf_script_path = os.path.join(_project_directory, "resources", _simulation_lsf_script_name)
# <--- MUDANÇA: Diretório de espectros H5 removido (não é mais usado)

# --- Configuração do Algoritmo Genético ---
population_size = 3
mutation_rate = 0.2
num_generations = 2

# --- Critério de Convergência ---
enable_convergence_check = True
CONVERGENCE_PATIENCE = int(num_generations * 0.2)
if CONVERGENCE_PATIENCE < 20:
    CONVERGENCE_PATIENCE = 20

# --- Enable de limpeza dos arquivos para debug ---
clean_enable = True

# <--- MUDANÇA: Ranges de Parâmetros (Novo cromossomo) ---
Lambda_range = (0.2e-6, 0.4e-6)
DC_range = (0.3, 0.7)
w_range = (0.4e-6, 0.6e-6)
# w_c_range é o limite absoluto. genetic.py aplica a restrição w_c < 0.8*w
w_c_range = (1e-7, w_range[1] * 0.8) 
N_range = (2, 500) # N é int, conforme definido em genetic.py

# Opções disponíveis: "delta_amp", "highpass", "lowpass", "bandpass"
FITNESS_STRATEGY_NAME = "delta_amp" # <--- Defina sua estratégia (ex: "delta_amp")

# --- Parâmetros para as Estratégias de Fitness (Filtros) ---
c = 299792458  # Velocidade da luz em m/s
CUTOFF_WAVELENGTH_NM = 1565
CENTER_WAVELENGTH_NM = 1550
BANDWIDTH_NM = 50
TRANSITION_BANDWIDTH_HZ = 5e12 # 5 THz
WEIGHT_REJECTION = 0.30
WEIGHT_PASSBAND = 0.30
WEIGHT_TRANSITION = 0.40

# --- Verificação de Sanidade dos Pesos ---
if not np.isclose(WEIGHT_REJECTION + WEIGHT_PASSBAND + WEIGHT_TRANSITION, 1.0):
    print("AVISO: A soma dos pesos da função de fitness não é 1.0! O fitness final não estará normalizado.")

# --- Instanciação da Estratégia de Fitness ---
fitness_calculator = None
print("--------------------------------------------------------------------------")
print(f"Configurando a otimização...")

if FITNESS_STRATEGY_NAME == "delta_amp":
    fitness_calculator = DeltaAmpStrategy()
    print(f"Estratégia selecionada: {fitness_calculator.__class__.__name__}")

elif FITNESS_STRATEGY_NAME == "highpass":
    print(f"Estratégia selecionada: HighpassStrategy")
    cutoff_wavelength_m = CUTOFF_WAVELENGTH_NM * 1e-9
    f_cutoff = c / cutoff_wavelength_m
    fitness_calculator = HighpassStrategy(
        f_cutoff=f_cutoff,
        transition_bandwidth=TRANSITION_BANDWIDTH_HZ,
        w_rejection=WEIGHT_REJECTION,
        w_passband=WEIGHT_PASSBAND,
        w_transition=WEIGHT_TRANSITION
    )
    print(f"--> Configuração: Corte em {CUTOFF_WAVELENGTH_NM} nm")

elif FITNESS_STRATEGY_NAME == "lowpass":
    print(f"Estratégia selecionada: LowpassStrategy")
    cutoff_wavelength_m = CUTOFF_WAVELENGTH_NM * 1e-9
    f_cutoff = c / cutoff_wavelength_m
    fitness_calculator = LowpassStrategy(
        f_cutoff=f_cutoff,
        transition_bandwidth=TRANSITION_BANDWIDTH_HZ,
        w_rejection=WEIGHT_REJECTION,
        w_passband=WEIGHT_PASSBAND,
        w_transition=WEIGHT_TRANSITION
    )
    print(f"--> Configuração: Corte em {CUTOFF_WAVELENGTH_NM} nm")

elif FITNESS_STRATEGY_NAME == "bandpass":
    print(f"Estratégia selecionada: BandpassStrategy")
    center_wavelength_m = CENTER_WAVELENGTH_NM * 1e-9
    f_center = c / center_wavelength_m
    f_upper = c / (center_wavelength_m - (BANDWIDTH_NM * 1e-9 / 2))
    f_lower = c / (center_wavelength_m + (BANDWIDTH_NM * 1e-9 / 2))
    bandwidth_hz = f_upper - f_lower
    fitness_calculator = BandpassStrategy(
        f_center=f_center,
        bandwidth=bandwidth_hz,
        transition_bandwidth=TRANSITION_BANDWIDTH_HZ,
        w_rejection=WEIGHT_REJECTION,
        w_passband=WEIGHT_PASSBAND,
        w_transition=WEIGHT_TRANSITION
    )
    print(f"--> Configuração: Canal de {BANDWIDTH_NM} nm centrado em {CENTER_WAVELENGTH_NM} nm")

else:
    raise ValueError(f"Estratégia de fitness '{FITNESS_STRATEGY_NAME}' é inválida.")

print("--------------------------------------------------------------------------")
print(f"Iniciando otimização com a estratégia: {fitness_calculator.__class__.__name__}")
print(f"Paciência para convergência: {CONVERGENCE_PATIENCE}")
print("--------------------------------------------------------------------------")

# <--- MUDANÇA: Copiando arquivo .lms
shutil.copy(_original_lms_path, _temp_lms_base_path)
print(f"Copiado {_original_lms_path} para {_temp_lms_base_path}")

if not os.path.exists(_temp_lms_base_path):
    raise FileNotFoundError(f"Erro: O arquivo base {_temp_lms_base_path} não foi criado.")

# <--- MUDANÇA: Inicialização do otimizador com novos ranges
optimizer = GeneticOptimizer(
    population_size, mutation_rate, num_generations,
    Lambda_range, DC_range, w_range, w_c_range, N_range
)
optimizer.initialize_population()
current_population = optimizer.population

experiment_start_time = datetime.datetime.now()
timestamp_str = experiment_start_time.strftime('%Y%m%d_%H%M%S')
full_data_csv_path = os.path.join(_simulation_results_directory, f"full_optimization_data_{timestamp_str}.csv")

generations_processed = 0
all_individuals_data = []
best_fitness_so_far = -float('inf')
generations_without_improvement = 0

try:
    # <--- MUDANÇA: Inicia o lumapi.MODE
    with lumapi.MODE(hide=False) as mode:
        for gen_num in range(num_generations):
            generations_processed += 1
            print(f"\n--- Processando Geração {gen_num + 1}/{num_generations} ---")
      
            # <--- MUDANÇA: Chamada da simulação EME
            # Retorna dados na memória, não caminhos de arquivo
            all_S_matrices_for_gen, frequencies = simulate_generation_lumerical(
                mode, current_population, _temp_lms_base_path,
                _geometry_lsf_script_path, _simulation_lsf_script_path,
                _temp_directory
            )
            
            print("\n  [Job Manager] Pós-processando os resultados da geração...")
            
            # <--- MUDANÇA: Cálculo de fitness a partir de dados na memória ---
            fitness_scores_for_gen = []
            
            # Se 'frequencies' for None, algo falhou catastroficamente (ex: Lumerical fechou)
            if frequencies is None:
                print("!!! Erro Crítico: 'frequencies' é None. A simulação pode ter falhado.")
                # Preenche todos os fitness com -inf para esta geração
                fitness_scores_for_gen = [-np.inf] * len(current_population)
            else:
                # Loop normal
                for S_matrix in all_S_matrices_for_gen:
                    # Se S_matrix for None, a simulação individual falhou.
                    if S_matrix is None:
                        fitness_scores_for_gen.append(-np.inf)
                        continue

                    # Se a simulação funcionou, calcula o fitness
                    try:
                        fitness_score = fitness_calculator.calculate(S_matrix, frequencies)
                    except Exception as e:
                        print(f"!!! Erro no cálculo do fitness para um indivíduo: {e}")
                        fitness_score = -np.inf
                    fitness_scores_for_gen.append(fitness_score)
            
            # Coleta de dados para o CSV
            for i, chromosome in enumerate(current_population):
                individual_data = chromosome.copy()
                # <--- MUDANÇA: Padronizado para 'Fitness' (para analysis.py)
                individual_data['Fitness'] = fitness_scores_for_gen[i]
                individual_data['generation'] = gen_num + 1
                individual_data['fitness_strategy'] = FITNESS_STRATEGY_NAME 
                all_individuals_data.append(individual_data)

            try:
                current_population = optimizer.evolve(fitness_scores_for_gen)
            except ValueError as e:
                print(f"!!! Erro na evolução da população: {e}")
                break

            print(f"  [Relatório] Atualizando relatório para a Geração {gen_num + 1}...")
            # <--- MUDANÇA: Passa os novos ranges para a função de relatório
            record_experiment_results(
                _simulation_results_directory, optimizer, experiment_start_time,
                Lambda_range, DC_range, w_range, w_c_range, N_range, 
                generations_processed
            )
            
            if all_individuals_data:
                df_all_data = pd.DataFrame(all_individuals_data)
                df_all_data.to_csv(full_data_csv_path, index=False)
                print(f"  [Análise] Dados de {len(all_individuals_data)} indivíduos atualizados em CSV.")
                # <--- MUDANÇA: analysis.py já está pronto para as novas colunas
                run_full_analysis(full_data_csv_path)
                print(f"  [Análise] Gráficos de análise atualizados e salvos.")

            # Lógica de convergência (inalterada)
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
    
    if clean_enable:
        # <--- MUDANÇA: Limpeza final
        clean_simulation_directory(_temp_directory, file_extension=".lms")
        clean_simulation_directory(_temp_directory, file_extension=".log")
        if os.path.exists(_temp_lms_base_path):
            os.remove(_temp_lms_base_path)
            print(f"\n[Limpeza Final] Arquivo base removido: {_temp_lms_base_path}")

except Exception as e:
    print(f"!!! Erro fatal no script principal de otimização: {e}")
    import traceback
    traceback.print_exc()

print("\nScript principal (main.py) finalizado.")