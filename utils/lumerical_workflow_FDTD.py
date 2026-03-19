# utils/lumerical_workflow_FDTD.py

import lumapi
import os
import numpy as np

def simulate_generation_lumerical(fdtd, population, temp_base_path,
                                  geom_lsf_path, simu_lsf_path, temp_dir, 
                                  mode_type="apodized", stop_check=None):
    """
    Simula a geração FDTD configurando as variáveis escalares do grupo
    e extraindo os S-parameters diretamente das Ports.
    """
    fdtd.clearjobs()
    job_files = []

    # 1. Carrega os scripts LSF uma única vez
    with open(geom_lsf_path, 'r', encoding='utf-8') as f:
        geom_script = f.read()
    with open(simu_lsf_path, 'r', encoding='utf-8') as f:
        simu_script = f.read()

    # 2. Preparação dos arquivos (Construção acelerada via LSF)
    for i, chromosome in enumerate(population):
        if stop_check and stop_check():
            return [], None
        
        file_name = f"indiv_{i+1}_fdtd.fsp"
        file_path = os.path.join(temp_dir, file_name)
        
        fdtd.load(temp_base_path)
        fdtd.switchtolayout()
        # Injeta os parâmetros diretamente nas propriedades do Structure Group
        # O script LSF ('geom_script') usará P, M e delta_s_max para desenhar os dentes
        fdtd.setnamed("Guia Metamaterial", "delta_s_max", float(chromosome['delta_s_max']))
        fdtd.setnamed("Guia Metamaterial", "P", float(chromosome['P']))
        fdtd.setnamed("Guia Metamaterial", "M", float(chromosome['M']))
        fdtd.setnamed("Guia Metamaterial", "Lambda", float(chromosome['Lambda']))
        fdtd.setnamed("Guia Metamaterial", "DC", float(chromosome['DC']))
        fdtd.setnamed("Guia Metamaterial", "w", float(chromosome['w']))
        fdtd.setnamed("Guia Metamaterial", "w_c", float(chromosome['w_c']))
        fdtd.setnamed("Guia Metamaterial", "N", int(chromosome['N']))

        # Executa a geometria e a configuração de simulação (Ports e Mesh)
        fdtd.eval(geom_script)
        fdtd.eval(simu_script) 
        
        fdtd.save(file_path)
        fdtd.addjob(file_path, "FDTD") 
        
        job_files.append(file_path)

    # 3. Execução Paralela
    print(f"      [FDTD] Executando runjobs() para {len(population)} indivíduos...")
    fdtd.runjobs()

    # 4. Coleta dos Espectros via Ports
    all_spectra = []
    frequencies = None

    for file_path in job_files:
        if stop_check and stop_check():
            return [], None
        try:
            fdtd.load(file_path)
            
            # --- Extração Apenas do Essencial: S11 (Reflexão) da Porta 1 ---
            res_port1 = fdtd.getresult("FDTD::ports::port_1", "S")
            f_vector = res_port1['f'].flatten()
            s11_complex = res_port1['S'].flatten()
            print(s11_complex)
            # --- Montagem da Matriz S ---
            num_pts = len(f_vector)
            S_equiv = np.zeros((2, 2, num_pts), dtype=np.complex128)
            
            # Injetamos a magnitude do S11 no índice [0, 0]
            # O restante da matriz continuará com zeros (S21, S12, S22 = 0)
            S_equiv[0, 0, :] = np.abs(s11_complex)
            
            all_spectra.append(S_equiv)
            
            if frequencies is None:
                frequencies = f_vector
                
        except Exception as e:
            print(f"      [Erro] Coleta falhou em {os.path.basename(file_path)}: {e}")
            all_spectra.append(None)

    return all_spectra, frequencies