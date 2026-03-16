# supervisor.py
# Script "mestre" para executar varreduras de parâmetros.
# Define a configuração, chama o 'main.py' em loop,
# gerencia a lógica de repetição e o estado da varredura.

import numpy as np
import copy
import datetime
import pandas as pd
import os
import json
import traceback

# Importa a função "trabalhadora" de main.py
try:
    from main import run_optimization
except ImportError as e:
    print(f"Erro: Não foi possível importar 'run_optimization' de main.py: {e}")
    print("Certifique-se de que main.py está no mesmo diretório e foi modularizado corretamente.")
    exit(1)

# --- [NOVO] Configuração Padrão ---
# O supervisor agora é o "dono" da configuração padrão.
DEFAULT_CONFIG = {
    "file_paths": {
        "original_lms_file_name": "SWG_period_EME.lms",
        "geometry_lsf_script_name": "create_guide_EME.lsf",
        "simulation_lsf_script_name": "run_simu_guide_EME.lsf",
        "simulation_results_directory_name": "simulation_results"
    },
    "ga_params": {
        "population_size": 50,
        "mutation_rate": 0.2,
        "num_generations": 160,
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
        # Estes são os padrões que serão substituídos pelo supervisor
        "strategy_name": "reflection_band",
        "cutoff_wl_nm": 1550,
        "center_wl_nm": 1550,
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


# --- PAINEL DE CONTROLE DA VARREDURA ---

# 1. Definições da Varredura
SWEEP_FITNESS_STRATEGY = "reflection_band"
# Faixa segura baseada na janela de simulação [1400, 1600] nm
WAVELENGTH_START_NM = 1450
WAVELENGTH_STOP_NM = 1550
WAVELENGTH_STEPS = 11

# 2. Lógica de Repetição (Retry)
BANDWIDTH_SWEEP_NM = [5, 10, 20]
FITNESS_THRESHOLD = 0.75

# 3. Arquivos de Gerenciamento do Supervisor (Salvos na pasta raiz)
SWEEP_STATE_FILE = "supervisor_state.json"
SWEEP_SUMMARY_FILE_PREFIX = "sweep_summary"

# --- Fim do Painel de Controle ---

def load_state_from_file(filename: str) -> dict:
    """Apenas lê o arquivo JSON ou retorna um estado padrão."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
                return state
        except Exception as e:
            print(f"!!! Erro ao ler o arquivo de estado {filename}: {e}. Reiniciando do zero.")
    
    return {"start_index": 0, "all_experiment_results": []}

def save_state(filename: str, state: dict):
    """Salva o estado atual da varredura em um arquivo JSON."""
    try:
        with open(filename, 'w') as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"!!! AVISO: Não foi possível salvar o estado da varredura em {filename}: {e}")

def run_sweep():
    """Executa a varredura completa de otimização."""
    print("--- SUPERVISOR: Iniciando varredura ---")
    start_time = datetime.datetime.now()
    
    wavelength_sweep_nm = np.linspace(WAVELENGTH_START_NM, WAVELENGTH_STOP_NM, WAVELENGTH_STEPS)
    total_planned_experiments = len(wavelength_sweep_nm)
    
    # Carrega o estado (arquivo na raiz)
    state = load_state_from_file(SWEEP_STATE_FILE)
    
    # Lógica de Auto-Reinicialização
    if state["start_index"] == total_planned_experiments:
        print(f"--- Varredura anterior concluída ({state['start_index']}/{total_planned_experiments} etapas). ---")
        print("--- REINICIANDO para uma nova varredura. ---")
        state = {"start_index": 0, "all_experiment_results": []}
        save_state(SWEEP_STATE_FILE, state)
    elif state["start_index"] > 0:
        print(f"--- Estado anterior encontrado. Retomando da etapa {state['start_index']} ---")
    
    all_results = state["all_experiment_results"]
    start_index = state["start_index"]

    try:
        # Loop principal da varredura de comprimento de onda
        for i, target_wl_nm in enumerate(wavelength_sweep_nm):
            
            # Lógica para RETOMAR
            if i < start_index:
                print(f"--- Pulando Etapa {i+1}/{total_planned_experiments} (WL={target_wl_nm:.1f} nm). Já concluída. ---")
                continue
            
            print(f"\n\n--- INICIANDO ETAPA {i+1}/{total_planned_experiments}: Alvo = {target_wl_nm:.1f} nm ---")
            
            best_fitness_for_this_wl = -np.inf
            best_config_for_this_wl = {}
            
            # Loop interno de REPETIÇÃO (Retry) por largura de banda
            for bw_nm in BANDWIDTH_SWEEP_NM:
                
                print(f"--- Tentativa com Largura de Banda: {bw_nm} nm ---")
                
                # 1. Cria uma cópia da configuração padrão
                current_config = copy.deepcopy(DEFAULT_CONFIG)
                
                # 2. Modifica a configuração para este experimento
                current_config['fitness_params']['strategy_name'] = SWEEP_FITNESS_STRATEGY
                current_config['fitness_params']['center_wl_nm'] = target_wl_nm
                current_config['fitness_params']['bandwidth_nm'] = bw_nm
                
                try:
                    # 3. Chama a função "trabalhadora"
                    best_fitness, csv_path = run_optimization(current_config)
                except Exception as e:
                    print(f"!!! Erro fatal na execução de 'run_optimization' para WL={target_wl_nm:.1f}, BW={bw_nm}: {e}")
                    traceback.print_exc()
                    best_fitness = -np.inf
                    csv_path = None # 'csv_path' é o caminho retornado por main.py

                # 4. Armazena o melhor resultado para este WL
                if best_fitness > best_fitness_for_this_wl:
                    best_fitness_for_this_wl = best_fitness
                    best_config_for_this_wl = {
                        "target_wavelength_nm": target_wl_nm,
                        "attempted_bandwidth_nm": bw_nm,
                        "best_fitness_achieved": best_fitness,
                        "results_csv_path": csv_path,
                        "status": "Success" if best_fitness >= FITNESS_THRESHOLD else "Fail"
                    }

                # 5. Verifica a condição de sucesso para parar as repetições
                if best_fitness >= FITNESS_THRESHOLD:
                    print(f"--- Sucesso! Fitness ({best_fitness:.4f}) atingiu o limiar ({FITNESS_THRESHOLD}). Passando para o próximo WL. ---")
                    break # Interrompe o loop de BW
                else:
                    print(f"--- Falha. Fitness ({best_fitness:.4f}) abaixo do limiar. Tentando com BW maior... ---")
            
            # Fim do loop de repetição
            
            # 6. Registra o melhor resultado obtido para este WL
            if not best_config_for_this_wl: # Se todas as tentativas falharam
                best_config_for_this_wl = {
                    "target_wavelength_nm": target_wl_nm,
                    "attempted_bandwidth_nm": "All",
                    "best_fitness_achieved": -np.inf,
                    "results_csv_path": None,
                    "status": "Error"
                }
            
            all_results.append(best_config_for_this_wl)
            
            # 7. SALVA O ESTADO (Tolerância a falhas)
            state["start_index"] = i + 1
            state["all_results"] = all_results
            save_state(SWEEP_STATE_FILE, state)

    except KeyboardInterrupt:
        print("\n--- Varredura interrompida pelo usuário. O progresso foi salvo e pode ser retomado. ---")
        return 

    # --- Finalização ---
    end_time = datetime.datetime.now()
    print("\n\n--- SUPERVISOR: Varredura Completa ---")
    print(f"Tempo total: {end_time - start_time}")

    results_df = pd.DataFrame(all_results)
    
    # Salva o arquivo de resumo na pasta raiz
    summary_path = f"{SWEEP_SUMMARY_FILE_PREFIX}_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(summary_path, index=False)
    
    print("Resultados da Varredura:")
    print(results_df)
    print(f"Resumo da varredura salvo em: {summary_path}")
    print(f"Arquivo de estado '{SWEEP_STATE_FILE}' mantido. A próxima execução será reiniciada automaticamente.")

if __name__ == "__main__":
    run_sweep()