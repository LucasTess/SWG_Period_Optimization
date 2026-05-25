# supervisor.py
import numpy as np
import copy
import datetime
import pandas as pd
import os
import glob
import json
import traceback

try:
    from main import run_optimization
except ImportError as e:
    print(f"Erro: Não foi possível importar 'run_optimization' de main.py: {e}")
    exit(1)

# --- [NOVO] Configuração Padrão ---
DEFAULT_CONFIG = {
    "file_paths": {
        "original_lms_file_name": "SWG_period_EME.lms",
        "geometry_lsf_script_name": "create_guide_EME.lsf",
        "simulation_lsf_script_name": "run_simu_guide_EME.lsf",
        "simulation_results_directory_name": "simulation_results"
    },
    "ga_params": {
        "population_size": 20,
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
FITNESS_THRESHOLD = 0.8

SWEEP_STATE_FILE = "supervisor_state.json"
SWEEP_SUMMARY_FILE_PREFIX = "sweep_summary"
# --- Fim do Painel de Controle ---

def load_state_from_file(filename: str) -> dict:
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"!!! Erro ao ler o arquivo de estado {filename}: {e}")
    return {"start_index": 0, "all_experiment_results": []}

def save_state(filename: str, state: dict):
    try:
        with open(filename, 'w') as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        pass

def check_for_checkpoint(target_wl, target_bw, config):
    """
    Vasculha a pasta de resultados por um experimento interrompido que 
    bata perfeitamente com as configurações atuais.
    Retorna (geracao_retomada, populacao_resgatada, path_csv_original) ou (0, None, None).
    """
    sim_dir = config['file_paths']['simulation_results_directory_name']
    if not os.path.exists(sim_dir):
        return 0, None, None

    csv_files = glob.glob(os.path.join(sim_dir, "*_full_data.csv"))
    pop_size = config['ga_params']['population_size']
    w_rej = config['fitness_params']['weights']['rejection']
    w_pass = config['fitness_params']['weights']['passband']
    w_trans = config['fitness_params']['weights']['transition']

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # 1. Verifica Integridade Básica
            required_cols = ['target_center_nm', 'target_bw_nm', 'w_rej', 'w_pass', 'w_trans', 'generation']
            if not all(c in df.columns for c in required_cols):
                continue
            
            # 2. Verifica a "Assinatura" do Experimento na primeira linha
            first_row = df.iloc[0]
            if not (abs(first_row['target_center_nm'] - target_wl) < 1e-3 and
                    abs(first_row['target_bw_nm'] - target_bw) < 1e-3 and
                    abs(first_row['w_rej'] - w_rej) < 1e-3 and
                    abs(first_row['w_pass'] - w_pass) < 1e-3 and
                    abs(first_row['w_trans'] - w_trans) < 1e-3):
                continue

            # É o nosso experimento! Vamos achar a última geração válida.
            gen_counts = df['generation'].value_counts()
            
            # Filtra apenas as gerações que tem tamanho == population_size
            valid_gens = [gen for gen, count in gen_counts.items() if count == pop_size]
            
            if not valid_gens:
                return 0, None, None # Nenhuma geração completou, começa do zero.
                
            last_valid_gen = max(valid_gens)
            
            # Extrai os cromossomos dessa última geração
            df_last_gen = df[df['generation'] == last_valid_gen]
            rescued_pop = []
            
            for _, row in df_last_gen.iterrows():
                chrom = {
                    'Lambda': float(row['Lambda']),
                    'DC': float(row['DC']),
                    'w': float(row['w']),
                    'w_c': float(row['w_c']),
                    'N': int(row['N'])
                }
                rescued_pop.append(chrom)
                
            return int(last_valid_gen), rescued_pop, file

        except Exception as e:
            continue

    return 0, None, None

def run_sweep():
    print("--- SUPERVISOR: Iniciando varredura ---")
    start_time = datetime.datetime.now()
    
    wavelength_sweep_nm = np.linspace(WAVELENGTH_START_NM, WAVELENGTH_STOP_NM, WAVELENGTH_STEPS)
    total_planned_experiments = len(wavelength_sweep_nm)
    
    state = load_state_from_file(SWEEP_STATE_FILE)
    
    if state["start_index"] == total_planned_experiments:
        print("--- REINICIANDO para uma nova varredura. ---")
        state = {"start_index": 0, "all_experiment_results": []}
        save_state(SWEEP_STATE_FILE, state)
    elif state["start_index"] > 0:
        print(f"--- Estado anterior encontrado. Retomando da etapa {state['start_index']} ---")
    
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
                
                # --- LÓGICA DE CHECKPOINT AQUI ---
                resumed_gen, resumed_pop, original_csv = check_for_checkpoint(target_wl_nm, bw_nm, current_config)
                
                if resumed_gen > 0:
                    print(f"  [CHECKPOINT ENCONTRADO] Retomando da Geração {resumed_gen} de um experimento anterior.")
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
            state["start_index"] = i + 1
            state["all_experiment_results"] = all_results
            save_state(SWEEP_STATE_FILE, state)

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