# utils/lumerical_workflow.py (MODIFICADO PARA EME e N variável)

import lumapi
import os
import shutil
import numpy as np

# --- Constantes da Simulação ---
# A altura é fixa por fabricação
HEIGHT_CONST = 0.22e-6
# Material do núcleo
MATERIAL_CONST = "Si (Silicon) - Palik" 
# REMOVIDO: N_PERIODS = 100 # N agora vem do cromossomo

# --- Configurações do EME Sweep (Controladas pelo Python) ---
WAVELENGTH_START = 1.4e-6
WAVELENGTH_STOP = 1.6e-6
WAVELENGTH_POINTS = 501
def _create_and_run_eme(mode, chromosome, lms_path, construction_lsf_path, simulation_lsf_path):
    """
    Função auxiliar que usa a API do MODE, executa os scripts LSF
    para construir/configurar o EME, executa o sweep e retorna a matriz S.
    """
    try:
        # 1. Carrega o arquivo base e limpa
        mode.load(lms_path)
        mode.switchtolayout()
        # (NÃO limpa, pois o LSF construtor assume que o grupo já existe)
        
        # 2. PYTHON CRIA O GRUPO E DEFINE AS PROPRIEDADES
        # Limpa simulações anteriores ANTES de criar o novo grupo
        mode.deleteall() 
        mode.addstructuregroup()
        mode.set("name", "Guia Metamaterial")
        
        mode.adduserprop("Lambda", 2, chromosome['Lambda'])
        mode.adduserprop("DC", 2, chromosome['DC'])
        mode.adduserprop("w", 2, chromosome['w'])
        mode.adduserprop("w_c", 2, chromosome['w_c'])
        mode.adduserprop("N", 2, chromosome['N']) 
        mode.adduserprop("height", 2, HEIGHT_CONST)
        mode.adduserprop("material", 5, MATERIAL_CONST)

        # 3. Roda o script LSF de construção (create_guide_EME.lsf)
        with open(construction_lsf_path, 'r') as f:
            create_lsf_content = f.read()
        mode.eval(create_lsf_content) # Este script NÃO deve ter 'deleteall'

        # 4. Roda o script de setup do EME (run_simu_guide_EME.lsf)
        with open(simulation_lsf_path, 'r') as f:
            simulate_lsf_content = f.read()
        mode.eval(simulate_lsf_content) # Este script define as portas e roda 'run'
        
        mode.save()

        # 5. PYTHON CONTROLA O SWEEP E EXECUTA
        mode.setemeanalysis("wavelength sweep", 1)
        mode.setemeanalysis("start wavelength", WAVELENGTH_START)
        mode.setemeanalysis("stop wavelength", WAVELENGTH_STOP)
        mode.setemeanalysis("number of wavelength points", WAVELENGTH_POINTS)
        
        print(f"    - Executando emesweep para {os.path.basename(lms_path)}...")
        mode.emesweep("wavelength sweep")
        
        # 6. PYTHON COLETA OS RESULTADOS
        print(f"    - Coletando resultados...")
        
        # --- [CORREÇÃO 1 AQUI] ---
        # O nome do resultado correto é "S_wavelength_sweep"
        S_matrix_dataset = mode.getemesweep("S_wavelength_sweep")
        
        # --- [CORREÇÃO 2 AQUI] ---
        # O resultado é um struct com 'wavelength' e 's11', 's12', etc.
        wavelengths = S_matrix_dataset['wavelength']
        # Converte comprimentos de onda (m) para frequências (Hz) para o fitness
        c = 299792458.0 
        frequencies = c / wavelengths
        
        num_freq = len(frequencies)
        S_matrix_3D = np.zeros((2, 2, num_freq), dtype=np.complex128)
        
        S_matrix_3D[0, 0, :] = S_matrix_dataset['s11']
        S_matrix_3D[0, 1, :] = S_matrix_dataset['s12']
        S_matrix_3D[1, 0, :] = S_matrix_dataset['s21']
        S_matrix_3D[1, 1, :] = S_matrix_dataset['s22']
        
        # Retorna a Matriz S e as FREQUÊNCIAS (que o fitness_functions.py espera)
        return S_matrix_3D, frequencies

    except Exception as e:
        print(f"!!! Erro durante a modificação/execução de {os.path.basename(lms_path)}: {e}")
        return None, None
    
def simulate_generation_lumerical(mode_session, current_population, lms_base_path,
                                  geometry_lsf_path, 
                                  simulation_lsf_path, temp_directory):
    """
    Executa UMA simulação EME para cada cromossomo na população.
    """
    all_S_matrices = []
    frequencies = None
    
    for chrom_id, chromosome in enumerate(current_population):
        print(f"\n--- Processando Cromossomo {chrom_id + 1}/{len(current_population)} ---")
        
        lms_main_path = os.path.join(temp_directory, f"chrom_{chrom_id+1}_eme.lms")

        shutil.copy(lms_base_path, lms_main_path)

        print(f"  - Modificando e executando {os.path.basename(lms_main_path)}...")
        S_total, current_frequencies = _create_and_run_eme(
            mode_session, 
            chromosome, 
            lms_main_path, 
            geometry_lsf_path, 
            simulation_lsf_path
        )
        
        all_S_matrices.append(S_total)
        if frequencies is None and current_frequencies is not None:
            frequencies = current_frequencies
            
    return all_S_matrices, frequencies