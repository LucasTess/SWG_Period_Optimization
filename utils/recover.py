# utils/recover.py
import os
import glob
import json
import pandas as pd
import re

def almost_equal(val1, val2, tol=1e-5):
    """Compara dois floats com uma tolerância para evitar erros de arredondamento."""
    return abs(float(val1) - float(val2)) < tol

def check_for_checkpoint(config: dict):
    """
    Vasculha a pasta de resultados por um experimento interrompido.
    """
    sim_dir = config['file_paths']['simulation_results_directory_name']
    if not os.path.exists(sim_dir):
        return 0, None, None, -float('inf')

    target_opt = config.get('optimizer_type', 'GA')
    target_pop = config['ga_params']['population_size']
    target_mut = config['ga_params']['mutation_rate'] if target_opt == 'GA' else None
    
    target_ranges = config['ga_ranges']
    expected_w_c_range = [1e-7, target_ranges['w_range'][1] * target_ranges['w_c_range_max_ratio']]
    
    fit_cfg = config['fitness_params']
    target_wl = fit_cfg['center_wl_nm']
    target_bw = fit_cfg['bandwidth_nm']
    target_trans = fit_cfg['transition_bw_nm']
    w_rej = fit_cfg['weights']['rejection']
    w_pass = fit_cfg['weights']['passband']
    w_trans = fit_cfg['weights']['transition']

    json_files = glob.glob(os.path.join(sim_dir, "results_*", "*.json"))
    
    for j_file in json_files:
        try:
            with open(j_file, 'r') as f:
                data = json.load(f)
            
            if data.get('optimizer_type', 'GA') != target_opt: continue
            if data.get('population_size') != target_pop: continue
            
            if target_mut is not None and 'mutation_rate' in data:
                if not almost_equal(data['mutation_rate'], target_mut): continue
            
            j_ranges = data.get('parameter_ranges', {})
            if not j_ranges: continue
            
            if not almost_equal(j_ranges.get('Lambda')[0], target_ranges['Lambda_range'][0]) or not almost_equal(j_ranges.get('Lambda')[1], target_ranges['Lambda_range'][1]): continue
            if not almost_equal(j_ranges.get('DC')[0], target_ranges['DC_range'][0]) or not almost_equal(j_ranges.get('DC')[1], target_ranges['DC_range'][1]): continue
            if not almost_equal(j_ranges.get('w')[0], target_ranges['w_range'][0]) or not almost_equal(j_ranges.get('w')[1], target_ranges['w_range'][1]): continue
            if j_ranges.get('N') != list(target_ranges['N_range']): continue
            
            j_w_c = j_ranges.get('w_c', [0, 0])
            if not (almost_equal(j_w_c[0], expected_w_c_range[0]) and almost_equal(j_w_c[1], expected_w_c_range[1])):
                continue
                
            j_fit = data.get('fitness_configuration', {})
            if not almost_equal(j_fit.get('center_wavelength_nm', 0), target_wl): continue
            if not almost_equal(j_fit.get('bandwidth_nm', 0), target_bw): continue
            if not almost_equal(j_fit.get('transition_bandwidth_nm', 0), target_trans): continue
            if not almost_equal(j_fit.get('weight_rejection', 0), w_rej): continue
            if not almost_equal(j_fit.get('weight_passband', 0), w_pass): continue
            if not almost_equal(j_fit.get('weight_transition', 0), w_trans): continue

            match = re.search(r'experiment_results_(\d{8}_\d{6})\.json', os.path.basename(j_file))
            if not match: continue
            
            ts = match.group(1)
            csv_pattern = os.path.join(sim_dir, f"*{ts}_full_data.csv")
            csv_matches = glob.glob(csv_pattern)
            
            if not csv_matches: continue
            csv_path = csv_matches[0]
            
            df = pd.read_csv(csv_path)
            gen_counts = df['generation'].value_counts()
            
            valid_gens = [gen for gen, count in gen_counts.items() if count == target_pop]
            if not valid_gens: continue 
                
            last_valid_gen = max(valid_gens)
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
                
            best_fitness = float(data.get('best_fitness_so_far', -float('inf')))
                
            return int(last_valid_gen), rescued_pop, csv_path, best_fitness

        except Exception:
            continue

    return 0, None, None, -float('inf')

# =========================================================================
# GESTÃO DE ESTADO DO SUPERVISOR
# =========================================================================
def load_or_create_sweep_state(filename: str, config: dict, sweep_params: dict) -> dict:
    """
    Gerencia o arquivo supervisor_state.json. Se a configuração de varredura mudar,
    ele reinicia o estado automaticamente.
    """
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            # Se os parâmetros do sweep ou o otimizador forem os mesmos, retomamos de onde parou.
            if state.get("sweep_params") == sweep_params and state.get("optimizer_type") == config.get("optimizer_type"):
                return state
            else:
                print("\n[*] Novos parâmetros de varredura ou otimizador detectados.")
                print("    -> Reiniciando o supervisor_state.json para a etapa 1.")
        except Exception as e:
            print(f"\n[!] Erro ao ler estado existente ({e}). Recriando arquivo limpo.")
            
    # Se o arquivo não existe, está vazio, ou as configs mudaram, cria um estado inicial
    initial_state = {
        "optimizer_type": config.get('optimizer_type', 'GA'),
        "sweep_params": sweep_params,
        "start_index": 0,
        "all_experiment_results": []
    }
    
    save_sweep_state(filename, initial_state)
    return initial_state

def save_sweep_state(filename: str, state: dict):
    try:
        with open(filename, 'w') as f:
            json.dump(state, f, indent=4)
    except Exception:
        pass