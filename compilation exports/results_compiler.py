import sys
import os
import glob
import pandas as pd
import numpy as np

# --- 1. Mapeamento de Caminhos e Injeção do Root ---
# Descobre a pasta raiz do projeto (voltando 1 nível a partir de compilation_exports)
_compiler_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_compiler_dir)

if _project_root not in sys.path:
    sys.path.append(_project_root)

# Caminhos vitais do projeto
_sim_results_dir = os.path.join(_project_root, "simulation_results")
_temp_dir = os.path.join(_project_root, "temp")
os.makedirs(_temp_dir, exist_ok=True)

# Arquivos base do EME
_base_file_path = os.path.join(_project_root, "bragg_guide_EME.lms")
_geom_script_path = os.path.join(_project_root, "resources", "create_guide_EME.lsf")
_sim_script_path = os.path.join(_project_root, "resources", "run_simu_guide_EME.lsf")

# --- 2. Caminho para a API do Lumerical ---
_lumapi_module_path = "C:\\Program Files\\Lumerical\\v241\\api\\python"
if _lumapi_module_path not in sys.path:
    sys.path.append(_lumapi_module_path)

import lumapi
from utils.lumerical_workflow_EME import simulate_generation_lumerical

def truncate_chromosome(row):
    """
    Aplica o truncamento exigido para refletir a precisão de fabricação:
    - w, w_c, Lambda: Notação científica com 2 casas decimais (ex: 3.26e-7)
    - DC: Arredondado para 2 casas decimais normais (ex: 0.21)
    - N: Inteiro absoluto
    """
    return {
        'Lambda': float(f"{row['Lambda']:.2e}"),
        'w':      float(f"{row['w']:.2e}"),
        'w_c':    float(f"{row['w_c']:.2e}"),
        'DC':     round(row['DC'], 2),
        'N':      int(row['N'])
    }

def compile_successful_spectra():
    print("\n==================================================================")
    print(" INICIANDO COMPILADOR DE RESULTADOS - GUIAS UNIFORMES (EME)")
    print("==================================================================\n")
    
    # 1. Vasculha TODOS os CSVs na pasta (sem restrição de sufixo)
    all_csvs = glob.glob(os.path.join(_sim_results_dir, "*.csv"))
    
    if not all_csvs:
        print(f"[ERRO] Nenhum arquivo '.csv' encontrado na pasta:\n{_sim_results_dir}")
        return

    population_to_simulate = []
    metadata_list = []
    
    print("[1/3] Analisando o DNA dos arquivos para isolar os guias uniformes...")
    
    for file in all_csvs:
        # Ignora arquivos que sabemos que não são de população (sumários)
        nome_arquivo = os.path.basename(file).lower()
        if "sweep_summary" in nome_arquivo or "results_s11" in nome_arquivo:
            continue
            
        try:
            df = pd.read_csv(file)
            colunas_lower = [c.lower() for c in df.columns]
            
            # Verifica se é realmente um arquivo de população (tem que ter Lambda e DC)
            if 'lambda' not in colunas_lower or 'dc' not in colunas_lower:
                continue
            
            # FILTRO DE DNA: Se tem 'H' ou 'alpha', é apodizado (pula o arquivo)
            if 'h' in colunas_lower or 'alpha_param' in colunas_lower or 'alpha' in colunas_lower:
                continue
                
            # Se passou pelos filtros, é um arquivo do EME (Uniforme) válido!
            col_fit = next((c for c in df.columns if c.lower() == 'fitness'), 'Fitness')
            
            best_idx = df[col_fit].idxmax()
            best_row = df.loc[best_idx]
            best_fitness = best_row[col_fit]
            
            wl_col = next((c for c in df.columns if 'target_center' in c.lower() or 'wl' in c.lower()), None)
            bw_col = next((c for c in df.columns if 'target_bw' in c.lower() or 'bw' in c.lower()), None)
            
            wl_target = best_row[wl_col] if wl_col else 1500.0
            bw_target = best_row[bw_col] if bw_col else 5.0
            
            # Critério de Sucesso (>= 0.75)
            if best_fitness >= 0.75:
                truncated_chrom = truncate_chromosome(best_row)
                population_to_simulate.append(truncated_chrom)
                metadata_list.append({
                    'fitness': best_fitness,
                    'wl_target': wl_target,
                    'bw_target': bw_target
                })
        except Exception as e:
            # Silencia erros de leitura de planilhas aleatórias que possam estar na pasta
            pass

    if population_to_simulate:
        combined = sorted(zip(population_to_simulate, metadata_list), key=lambda x: x[1]['wl_target'])
        population_to_simulate = [item[0] for item in combined]
        metadata_list = [item[1] for item in combined]

    num_guides = len(population_to_simulate)
    print(f"\n=> {num_guides} guias EME bem-sucedidos encontrados e truncados para fabricação.")
    
    if num_guides == 0:
        print("[AVISO] Nenhum guia uniforme atendeu aos critérios ou os arquivos não foram encontrados.")
        return

    # 2. Motor Lumerical EME
    print("\n[2/3] Iniciando Lumerical MODE API e enviando população truncada...")
    try:
        session = lumapi.MODE(hide=False)
        all_S_matrices, frequencies = simulate_generation_lumerical(
            mode_session=session,
            current_population=population_to_simulate, # <-- CORREÇÃO AQUI
            lms_base_path=_base_file_path,             # <-- CORREÇÃO AQUI (era base_file_path)
            geometry_lsf_path=_geom_script_path,       # <-- CORREÇÃO AQUI (era geometry_script_path)
            simulation_lsf_path=_sim_script_path,      # <-- CORREÇÃO AQUI (era simulation_script_path)
            temp_directory=_temp_dir,                  # <-- CORREÇÃO AQUI (era temp_dir)
            mode_type="uniform"
        )
        session.close()
    except Exception as e:
        print(f"\n[ERRO FATAL] Falha no Lumerical: {e}")
        return

    if frequencies is None or not all_S_matrices:
        print("\n[ERRO] O Lumerical não retornou matrizes S.")
        return

    # 3. Extração e Salvamento
    print("\n[3/3] Extraindo espectros S11 e montando a planilha de resultados...")
    c = 299792458
    wavelengths_nm = (c / frequencies) * 1e9
    
    df_spectra = pd.DataFrame({'Wavelength_nm': wavelengths_nm.flatten()})

    for idx, (S_matrix, meta) in enumerate(zip(all_S_matrices, metadata_list)):
        if S_matrix is not None:
            # S11 Power Reflectance
            R_spectrum = np.abs(S_matrix[0, 0, :]) ** 2
            col_name = f"Fit_{meta['fitness']:.4f}_WL_{meta['wl_target']}nm_BW_{meta['bw_target']}nm"
            df_spectra[col_name] = R_spectrum

    output_path = os.path.join(_compiler_dir, "results_S11_spectra.csv")
    df_spectra.to_csv(output_path, index=False)
    
    print("\n==================================================================")
    print(f" SUCESSO! Espectros salvos em:\n {output_path}")
    print("==================================================================\n")
if __name__ == "__main__":
    compile_successful_spectra()