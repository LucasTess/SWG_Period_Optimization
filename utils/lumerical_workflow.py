# utils/lumerical_workflow.py (Controle Total pelo Python)

import lumapi
import os
import shutil
import numpy as np

def _create_and_run_fsp(fdtd, chromosome, fsp_path, construction_lsf_path, simulation_lsf_path):
    """
    Função auxiliar que cria a estrutura via API Python, executa os scripts LSF
    para construir e simular, e retorna a matriz S.
    """
    try:
        # 1. Carrega o arquivo base (que contém materiais, etc.) e limpa
        fdtd.load(fsp_path)
        fdtd.switchtolayout()
        fdtd.deleteall()

        # 2. PYTHON CRIA O GRUPO E DEFINE AS PROPRIEDADES DO CROMOSSOMO
        fdtd.addstructuregroup()
        fdtd.set("name", "Guia Metamaterial")
        # Define as propriedades com os valores EXATOS do cromossomo
        fdtd.adduserprop("Lambda", 2, chromosome['Lambda'])
        fdtd.adduserprop("DC", 2, chromosome['DC'])
        fdtd.adduserprop("w", 2, chromosome['w'])
        fdtd.adduserprop("height", 2, chromosome['height'])
        fdtd.adduserprop("material", 5, 'Si (Silicon) - Palik') # Adiciona outras props se necessário

        # 3. Roda o script LSF de construção, que agora apenas lê os valores e desenha
        with open(construction_lsf_path, 'r') as f:
            create_lsf_content = f.read()
        fdtd.eval(create_lsf_content)

        # 4. Roda o script de setup da simulação
        with open(simulation_lsf_path, 'r') as f:
            simulate_lsf_content = f.read()
        fdtd.eval(simulate_lsf_content)
        
        # 5. Salva e executa
        fdtd.save()
        fdtd.runsweep("s-parameter sweep")
        S_matrix_dataset = fdtd.getsweepresult("s-parameter sweep", "S matrix")
        return S_matrix_dataset['S'], S_matrix_dataset['f']

    except Exception as e:
        print(f"!!! Erro durante a modificação/execução de {os.path.basename(fsp_path)}: {e}")
        return None, None

def simulate_generation_lumerical(fdtd, current_population, fsp_base_path,
                                  geometry_lsf_path, reference_lsf_path, 
                                  simulation_lsf_path, temp_directory):
    S_matrixes_raw = []
    S_matrixes_ref = []
    frequencies = None
    
    for chrom_id, chromosome in enumerate(current_population):
        print(f"\n--- Processando Cromossomo {chrom_id + 1}/{len(current_population)} ---")
        
        fsp_main_path = os.path.join(temp_directory, f"chrom_{chrom_id+1}_main.fsp")
        fsp_ref_path = os.path.join(temp_directory, f"chrom_{chrom_id+1}_ref.fsp")

        shutil.copy(fsp_base_path, fsp_main_path)
        shutil.copy(fsp_base_path, fsp_ref_path)

        print(f"  - Modificando e executando {os.path.basename(fsp_main_path)}...")
        S_raw, current_frequencies = _create_and_run_fsp(
            fdtd, chromosome, fsp_main_path, geometry_lsf_path, simulation_lsf_path
        )
        
        print(f"  - Modificando e executando {os.path.basename(fsp_ref_path)}...")
        S_ref, _ = _create_and_run_fsp(
            fdtd, chromosome, fsp_ref_path, reference_lsf_path, simulation_lsf_path
        )

        S_matrixes_raw.append(S_raw)
        S_matrixes_ref.append(S_ref)
        if frequencies is None and current_frequencies is not None:
            frequencies = current_frequencies
            
    return S_matrixes_raw, S_matrixes_ref, frequencies