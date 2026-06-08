# supervisor.py
import numpy as np
import copy
import datetime
import pandas as pd
import os

try:
    from main import run_optimization
except ImportError as e:
    print(f"Erro: Não foi possível importar 'run_optimization' de main.py: {e}")
    exit(1)
    
# Importa o módulo isolado de resgate e gestão de estado
from utils.recover import check_for_checkpoint, load_or_create_sweep_state, save_sweep_state

# --- Configuração Padrão ---
DEFAULT_CONFIG = {
    "optimizer_type": "WOA",  # Opções: "WOA" ou "GA"
    "file_paths": {
        "original_lms_file_name": "SWG_period_EME.lms",
        "geometry_lsf_script_name": "create_guide_EME.lsf",
        "simulation_lsf_script_name": "run_simu_guide_EME.lsf",
        "simulation_results_directory_name": "simulation_results"
    },
    "ga_params": {
        "population_size": 60,
        "mutation_rate": 0.2,
        "num_generations": 300,
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
        "cutoff_wl_nm": 1550,
        "center_wl_nm": 1550,
        "bandwidth_nm": 5,
        "transition_bw_nm": 20,
        "weights": {
            "rejection": 0.20,
            "passband": 0.60,
            "transition": 0.20
        }
    },
    "run_settings": {
        "clean_temp_files": True,
        "lumerical_hide_ui": True
    }
}

# --- PAINEL DE CONTROLE DA VARREDURA ---
SWEEP_FITNESS_STRATEGY = "reflection_band"
WAVELENGTH_START_NM = 1450
WAVELENGTH_STOP_NM = 1550
WAVELENGTH_STEPS = 11

BANDWIDTH_SWEEP_NM = [8,10,12]
FITNESS_THRESHOLD = 0.75

SWEEP_STATE_FILE = "supervisor_state.json"
SWEEP_SUMMARY_FILE_PREFIX = "sweep_summary"
# --- Fim do Painel de Controle ---

def run_sweep():
    print("--- SUPERVISOR: Iniciando varredura ---")
    start_time = datetime.datetime.now()
    
    wavelength_sweep_nm = np.linspace(WAVELENGTH_START_NM, WAVELENGTH_STOP_NM, WAVELENGTH_STEPS)
    total_planned_experiments = len(wavelength_sweep_nm)
    
# Agrupa os parâmetros para garantir a integridade absoluta do estado
    sweep_params = {
        "strategy": SWEEP_FITNESS_STRATEGY,
        "start_nm": WAVELENGTH_START_NM,
        "stop_nm": WAVELENGTH_STOP_NM,
        "steps": WAVELENGTH_STEPS,
        "bandwidths": BANDWIDTH_SWEEP_NM,
        "threshold": FITNESS_THRESHOLD,
        # Salva a geometria para detectar mudanças nos ranges
        "ranges": DEFAULT_CONFIG["ga_ranges"],
        # Salva os pesos para detectar mudanças nas prioridades da otimização
        "weights": DEFAULT_CONFIG["fitness_params"]["weights"]
    }
    
    # Delegamos a gestão de estado inicial para o recover.py
    state = load_or_create_sweep_state(SWEEP_STATE_FILE, DEFAULT_CONFIG, sweep_params)
    
    if state["start_index"] == total_planned_experiments:
        print("--- Varredura anterior já estava 100% concluída. REINICIANDO para uma nova. ---")
        state["start_index"] = 0
        state["all_experiment_results"] = []
        save_sweep_state(SWEEP_STATE_FILE, state)
    elif state["start_index"] > 0:
        print(f"--- Estado anterior encontrado e válido. Retomando da etapa {state['start_index']} ---")
    
    all_results = state["all_experiment_results"]
    start_index = state["start_index"]

    try:
        for i, target_wl_nm in enumerate(wavelength_sweep_nm):
            if i < start_index:
                print(f"--- Pulando Etapa {i+1} (WL={target_wl_nm:.1f} nm). ---")
                continue
            
            print(f"\n\n--- INICIANDO ETAPA {i+1}/{total_planned_experiments}: Alvo = {target_wl_nm:.1f} nm ---")
            
            best_fitness_for_this_wl = -np.inf
            best_config_for_this_wl = {}
            
            for bw_nm in BANDWIDTH_SWEEP_NM:
                print(f"--- Tentativa com Largura de Banda: {bw_nm} nm ---")
                
                current_config = copy.deepcopy(DEFAULT_CONFIG)
                current_config['fitness_params']['strategy_name'] = SWEEP_FITNESS_STRATEGY
                current_config['fitness_params']['center_wl_nm'] = target_wl_nm
                current_config['fitness_params']['bandwidth_nm'] = bw_nm
                
                resumed_gen, resumed_pop, original_csv, past_best_fitness = check_for_checkpoint(current_config)
                
                is_completed = (resumed_gen == current_config['ga_params']['num_generations'])
                is_successful = (past_best_fitness >= FITNESS_THRESHOLD)
                
                if resumed_gen > 0 and (is_completed or is_successful):
                    print(f"  [RESGATE RÁPIDO] Experimento concluído (Gen {resumed_gen}, Fit {past_best_fitness:.4f}). Pulando Lumerical.")
                    best_fitness = past_best_fitness
                    csv_path = original_csv
                else:
                    if resumed_gen > 0:
                        print(f"  [CHECKPOINT ENCONTRADO] Retomando da Geração {resumed_gen} de um experimento anterior.")
                        current_config['checkpoint'] = {
                            'start_generation': resumed_gen,
                            'population': resumed_pop,
                            'original_csv_path': original_csv
                        }
                    
                    try:
                        best_fitness, csv_path = run_optimization(current_config)
                    except Exception as e:
                        print(f"!!! Erro fatal em 'run_optimization': {e}")
                        best_fitness = -np.inf
                        csv_path = None

                if best_fitness > best_fitness_for_this_wl:
                    best_fitness_for_this_wl = best_fitness
                    best_config_for_this_wl = {
                        "target_wavelength_nm": target_wl_nm,
                        "attempted_bandwidth_nm": bw_nm,
                        "best_fitness_achieved": best_fitness,
                        "results_csv_path": csv_path,
                        "status": "Success" if best_fitness >= FITNESS_THRESHOLD else "Fail"
                    }

                if best_fitness >= FITNESS_THRESHOLD:
                    print(f"--- Sucesso! Passando para o próximo WL. ---")
                    break
                else:
                    print(f"--- Falha. Tentando com BW maior... ---")
            
            if not best_config_for_this_wl:
                best_config_for_this_wl = {
                    "target_wavelength_nm": target_wl_nm,
                    "attempted_bandwidth_nm": "All",
                    "best_fitness_achieved": -np.inf,
                    "results_csv_path": None,
                    "status": "Error"
                }
            
            all_results.append(best_config_for_this_wl)
            
            # --- SALVA O ESTADO A CADA PASSO ---
            state["start_index"] = i + 1
            state["all_experiment_results"] = all_results
            save_sweep_state(SWEEP_STATE_FILE, state)

    except KeyboardInterrupt:
        print("\n--- Varredura interrompida pelo usuário. ---")
        return 

    end_time = datetime.datetime.now()
    print("\n\n--- SUPERVISOR: Varredura Completa ---")
    results_df = pd.DataFrame(all_results)
    summary_path = f"{SWEEP_SUMMARY_FILE_PREFIX}_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(summary_path, index=False)
    print("Resultados da Varredura:")
    print(results_df)

if __name__ == "__main__":
    run_sweep()