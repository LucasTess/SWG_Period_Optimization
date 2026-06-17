# utils/lumerical_workflow.py
import sys
import os
import shutil
import numpy as np
import time
import concurrent.futures
import gc

_lumapi_module_path = "C:\\Program Files\\Lumerical\\v241\\api\\python"
if _lumapi_module_path not in sys.path:
    sys.path.append(_lumapi_module_path)

import lumapi

# ==============================================================================
# Se a máquina chorar, baixe isto para 6 ou 8.
MAX_LUMERICAL_WORKERS = 10
# ==============================================================================

def worker_simulate_chunk(worker_id, chunk_indices, population, center_wl_m, bw_m,
                          base_lms_path, temp_dir, hide_ui=True):
    results = {}
    mode = None
    
    worker_dir = os.path.join(temp_dir, f"worker_{worker_id}")
    os.makedirs(worker_dir, exist_ok=True)
    worker_dir_lum = worker_dir.replace('\\', '/')
    
    time.sleep(worker_id * 2.0) 
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            mode = lumapi.MODE(hide=hide_ui)
            break
        except Exception as e:
            wait_time = 5 + (attempt * 5)
            print(f"  [Worker {worker_id}] RAM/Licença ocupada. Aguardando {wait_time}s... (Tentativa {attempt+1}/{max_retries})")
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                print(f"❌ [Worker {worker_id}] Falha definitiva ao obter licença: {e}")
                return {idx: (None, None) for idx in chunk_indices}
                
    try:
        wl_start = center_wl_m - (bw_m * 2)
        wl_stop = center_wl_m + (bw_m * 2)

        for i in chunk_indices:
            chrom = population[i]
            unique_lms = os.path.join(worker_dir, f"sim_ind_{i}.lms")
            
            shutil.copy(base_lms_path, unique_lms)
            
            mode.load(unique_lms.replace('\\', '/'))
            mode.eval(f"cd('{worker_dir_lum}');")
            
            lsf_code = f"""
switchtolayout;
selectall; delete;

# Variaveis
Lambda = {float(chrom['Lambda'])};
DC = {float(chrom['DC'])};
w = {float(chrom['w'])};
w_c = {float(chrom['w_c'])};
N = {int(chrom['N'])};
height = 0.22e-6;
mat_sub = "SiO2 (Glass) - Palik";
mat_core = "Si (Silicon) - Palik";

# Geometria
addrect; set("name", "Sub_inf"); set("material", mat_sub);
set("x", 0); set("x span", Lambda*8); set("y", 0); set("y span", w*3);
set("z", -0.22e-6); set("z span", 0.22e-6);

addrect; set("name", "Sub_sup"); set("material", mat_sub);
set("x", 0); set("x span", Lambda*8); set("y", 0); set("y span", w*3);
set("z", 0); set("z span", height*3); set("alpha", 0.4);

addrect; set("name", "G_in"); set("material", mat_core);
set("x min", -2*Lambda); set("x max", -Lambda/2); set("y", 0); set("y span", w);
set("z", -0.11e-6 + height/2); set("z span", height);

addrect; set("name", "S_per"); set("material", mat_core);
set("x min", -Lambda/2); set("x max", -Lambda/2 + Lambda*DC); set("y", 0); set("y span", w);
set("z", -0.11e-6 + height/2); set("z span", height);

addrect; set("name", "S_core"); set("material", mat_core);
set("x min", -Lambda/2 + Lambda*DC); set("x max", Lambda/2); set("y", 0); set("y span", w_c);
set("z", -0.11e-6 + height/2); set("z span", height);

addrect; set("name", "G_out"); set("material", mat_core);
set("x min", Lambda/2); set("x max", Lambda*2); set("y", 0); set("y span", w);
set("z", -0.11e-6 + height/2); set("z span", height);

# Solver
addeme;
set("x min", -2*Lambda); set("y", 0); set("y span", w*1.5);
set("z", 0); set("z span", height*3);
select("EME");
set("display cells", 1);
set("number of cell groups", 4);
set("group spans", [(2*Lambda)-Lambda/2; Lambda*DC; Lambda*(1-DC); 2*Lambda-Lambda/2]);
set("cells", [1; 1; 1; 1]);
set("number of periodic groups", 1);
set("start cell group", [2]); set("end cell group", [3]); set("periods", N);

run;

# Sweep
setemeanalysis("wavelength sweep", 1);
setemeanalysis("start wavelength", {wl_start});
setemeanalysis("stop wavelength", {wl_stop});
setemeanalysis("number of wavelength points", 150);
emesweep("wavelength sweep");
"""
            try:
                mode.eval(lsf_code)
                
                S_matrix_dataset = mode.getemesweep("S_wavelength_sweep")
                wavelengths = S_matrix_dataset['wavelength'].flatten() 
                
                c_const = 299792458.0 
                freq = c_const / wavelengths
                num_freq = len(freq)
                
                S_matrix_3D = np.zeros((2, 2, num_freq), dtype=np.complex128)
                S_matrix_3D[0, 0, :] = S_matrix_dataset['s11'].flatten()
                S_matrix_3D[0, 1, :] = S_matrix_dataset['s12'].flatten()
                S_matrix_3D[1, 0, :] = S_matrix_dataset['s21'].flatten()
                S_matrix_3D[1, 1, :] = S_matrix_dataset['s22'].flatten()
                
                results[i] = (S_matrix_3D, freq)
                
            except Exception as e:
                results[i] = (None, None)
                
    finally:
        if mode:
            try: mode.close()
            except: pass
            del mode 
        
        gc.collect()

        try:
            for filename in os.listdir(worker_dir):
                file_path = os.path.join(worker_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path) 
        except:
            pass
            
    return results

def simulate_generation_lumerical(population, center_wl_m, bw_m, base_lms_path, temp_dir, hide_ui=True):
    pop_size = len(population)
    all_S_matrices = [None] * pop_size
    all_frequencies = None
    
    indices = list(range(pop_size))
    chunks = [list(c) for c in np.array_split(indices, min(MAX_LUMERICAL_WORKERS, pop_size))]
    
    print(f"  -> Distribuindo {pop_size} indivíduos em {len(chunks)} sessões paralelas...")

    # --- PROTEÇÃO NÍVEL SO: Limpeza Pré-Geração ---
    try:
        os.system("taskkill /F /IM mode.exe >nul 2>&1")
        os.system("taskkill /F /IM eme-engine-msmpi.exe >nul 2>&1")
    except:
        pass

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_LUMERICAL_WORKERS) as executor:
        for worker_id, chunk in enumerate(chunks):
            if len(chunk) > 0:
                futures.append(executor.submit(worker_simulate_chunk, worker_id, chunk, population, center_wl_m, bw_m, base_lms_path, temp_dir, hide_ui))
                
        for future in concurrent.futures.as_completed(futures):
            chunk_results = future.result()
            for idx, (S_mat, freq) in chunk_results.items():
                all_S_matrices[idx] = S_mat
                if freq is not None and all_frequencies is None:
                    all_frequencies = freq 

    # --- PROTEÇÃO NÍVEL SO: Limpeza Pós-Geração ---
    try:
        os.system("taskkill /F /IM mode.exe >nul 2>&1")
        os.system("taskkill /F /IM eme-engine-msmpi.exe >nul 2>&1")
    except:
        pass

    return all_S_matrices, all_frequencies