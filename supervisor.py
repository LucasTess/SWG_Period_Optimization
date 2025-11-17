# supervisor.py
# Script "mestre" para executar varreduras de parâmetros.
# [CORRIGIDO] Salva seus próprios arquivos de estado e resumo
# na pasta raiz, separado dos resultados dos experimentos.

import numpy as np
import copy
import datetime
import pandas as pd
import os
import json
import traceback

# Importa a função "trabalhadora" e a configuração padrão
try:
    from main import run_optimization, DEFAULT_CONFIG
except ImportError as e:
    print(f"Erro: Não foi possível importar 'run_optimization' de main.py: {e}")
    print("Certifique-se de que main.py está no mesmo diretório.")
    exit(1)

# --- PAINEL DE CONTROLE DA VARREDURA ---

# 1. Definições da Varredura
SWEEP_FITNESS_STRATEGY = "reflection_band"
WAVELENGTH_START_NM = 1450
WAVELENGTH_STOP_NM = 1550
WAVELENGTH_STEPS = 11

# 2. Lógica de Repetição (Retry)
BANDWIDTH_SWEEP_NM = [5, 10, 20]
FITNESS_THRESHOLD = 0.75

# 3. Arquivos de Gerenciamento do Supervisor (Salvos na pasta raiz)
SWEEP_STATE_FILE = "supervisor_state.json"
SWEEP_SUMMARY_FILE_PREFIX = "sweep_summary"

# Diretório onde os *experimentos* salvarão seus dados
RESULTS_DIR_BASE = DEFAULT_CONFIG['file_paths']['simulation_results_directory_name']

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
    
    # [LOGICA DE ESTADO] Carrega o estado (arquivo na raiz)
    state = load_state_from_file(SWEEP_STATE_FILE)
    
    if state["start_index"] == total_planned_experiments:
        print(f"--- Varredura anterior concluída ({state['start_index']}/{total_planned_experiments} etapas). ---")
        print("--- REINICIANDO para uma nova varredura. ---")
        state = {"start_index": 0, "all_experiment_results": []}
        save_state(SWEEP_STATE_FILE, state) # Salva o estado resetado
    elif state["start_index"] > 0:
        print(f"--- Estado anterior encontrado. Retomando da etapa {state['start_index']} ---")
    
    all_results = state["all_experiment_results"]
    start_index = state["start_index"]

    try:
        for i, target_wl_nm in enumerate(wavelength_sweep_nm):
            
            if i < start_index:
                print(f"--- Pulando Etapa {i+1}/{total_planned_experiments} (WL={target_wl_nm:.1f} nm). Já concluída. ---")
                continue
            
            print(f"\n\n--- INICIANDO ETAPA {i+1}/{total_planned_experiments}: Alvo = {target_wl_nm:.1f} nm ---")
            
            best_fitness_for_this_wl = -np.inf
            best_config_for_this_wl = {}
            
            for bw_nm in BANDWIDTH_SWEEP_NM:
                
                print(f"--- Tentativa com Largura de Banda: {bw_nm} nm ---")
                
                current_config = copy.deepcopy(DEFAULT_CONFIG)
                
                # Configura os parâmetros do experimento
                current_config['fitness_params']['strategy_name'] = SWEEP_FITNESS_STRATEGY
                current_config['fitness_params']['center_wl_nm'] = target_wl_nm
                current_config['fitness_params']['bandwidth_nm'] = bw_nm
                
                # [IMPORTANTE] Diz ao 'main.py' onde salvar os resultados *deste* experimento
                # Ex: 'simulation_results/1430.0nm_target'
                experiment_output_dir = os.path.join(
                    RESULTS_DIR_BASE, 
                    f"{target_wl_nm:.1f}nm_target"
                )
                current_config['file_paths']['simulation_results_directory_name'] = experiment_output_dir

                try:
                    # O 'main.py' agora salva seus próprios arquivos dentro do 'experiment_output_dir'
                    best_fitness, csv_path = run_optimization(current_config)
                except Exception as e:
                    print(f"!!! Erro fatal na execução de 'run_optimization' para WL={target_wl_nm:.1f}, BW={bw_nm}: {e}")
                    traceback.print_exc()
                    best_fitness = -np.inf
                    csv_path = None # 'csv_path' agora é o caminho completo para o CSV do experimento

                # ... (lógica de seleção de melhor fitness e parada, inalterada) ...
                if best_fitness > best_fitness_for_this_wl:
                    best_fitness_for_this_wl = best_fitness
                    best_config_for_this_wl = {
                        "target_wavelength_nm": target_wl_nm,
                        "achieved_bandwidth_nm": bw_nm,
                        "best_fitness_achieved": best_fitness,
                        "results_csv_path": csv_path, # Salva o caminho para o CSV de dados brutos
                        "status": "Success" if best_fitness >= FITNESS_THRESHOLD else "Fail"
                    }

                if best_fitness >= FITNESS_THRESHOLD:
                    print(f"--- Sucesso! Fitness ({best_fitness:.4f}) atingiu o limiar ({FITNESS_THRESHOLD}). Passando para o próximo WL. ---")
                    break
                else:
                    print(f"--- Falha. Fitness ({best_fitness:.4f}) abaixo do limiar. Tentando com BW maior... ---")
            
            if not best_config_for_this_wl: 
                best_config_for_this_wl = {
                    "target_wavelength_nm": target_wl_nm,
                    "achieved_bandwidth_nm": "N/A",
                    "best_fitness_achieved": -np.inf,
                    "results_csv_path": None,
                    "status": "Error"
                }
            
            all_results.append(best_config_for_this_wl)
            
            # Salva o estado (JSON na raiz)
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
    
    # [CORRIGIDO] Salva o arquivo de resumo na pasta raiz (junto com o supervisor.py)
    summary_path = f"{SWEEP_SUMMARY_FILE_PREFIX}_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
    
    results_df.to_csv(summary_path, index=False)
    
    print("Resultados da Varredura:")
    print(results_df)
    print(f"Resumo da varredura salvo em: {summary_path}")
    print(f"Arquivo de estado '{SWEEP_STATE_FILE}' mantido. A próxima execução será reiniciada automaticamente.")

if __name__ == "__main__":
    run_sweep()