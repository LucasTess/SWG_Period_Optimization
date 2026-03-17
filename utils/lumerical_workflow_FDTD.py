import lumapi
import os
import shutil
import numpy as np

# --- Configurações de Simulação FDTD ---
FDTD_WAVELENGTH_START = 1.4e-6
FDTD_WAVELENGTH_STOP = 1.6e-6
FDTD_WAVELENGTH_POINTS = 201

def _build_apodized_structure(fdtd, chromosome):
    """
    Constrói a geometria do guia SWG com apodização lateral.
    A força do acoplamento é modulada pelo deslocamento lateral (delta_s).
    """
    fdtd.switchtolayout()
    fdtd.deleteall()
    
    # Parâmetros do cromossomo
    W = float(chromosome['w'])
    W_C = float(chromosome['w_c'])
    LAMBDA = float(chromosome['Lambda'])
    N = int(chromosome['N'])
    DC = float(chromosome['DC'])
    
    # Parâmetros de Apodização Estocástica
    DS_MAX = float(chromosome['delta_s_max'])
    P = float(chromosome['P'])
    M = float(chromosome['M'])
    
    height = 0.22e-6
    L_total = N * LAMBDA
    
    # Construção da Estrutura (Período por Período)
    for n in range(1, N + 1):
        # Fórmula de Apodização Lateral discutida:
        # delta_s(n) define o desalinhamento entre as paredes laterais [cite: 40, 67, 117]
        x_norm = abs((2.0 * n - N) / N)
        delta_s = DS_MAX * (1.0 - x_norm**P)**M
        
        z_pos = (n - 0.5) * LAMBDA - (L_total / 2)
        
        # Dente Superior (Paredes laterais deslocadas)
        fdtd.addrect()
        fdtd.set("name", f"tooth_top_{n}")
        fdtd.set("x", 0)
        fdtd.set("x span", W + W_C)
        fdtd.set("y", z_pos + (delta_s / 2))
        fdtd.set("y span", LAMBDA * DC)
        fdtd.set("z", 0)
        fdtd.set("z span", height)
        fdtd.set("material", "Si (Silicon) - Palik")
        
        # Dente Inferior
        fdtd.addrect()
        fdtd.set("name", f"tooth_bottom_{n}")
        fdtd.set("x", 0)
        fdtd.set("x span", W + W_C)
        fdtd.set("y", z_pos - (delta_s / 2))
        fdtd.set("y span", LAMBDA * DC)
        fdtd.set("z", 0)
        fdtd.set("z span", height)
        fdtd.set("material", "Si (Silicon) - Palik")

    # Núcleo do guia (Base)
    fdtd.addrect()
    fdtd.set("name", "waveguide_core")
    fdtd.set("x", 0)
    fdtd.set("x span", W)
    fdtd.set("y", 0)
    fdtd.set("y span", L_total)
    fdtd.set("z", 0)
    fdtd.set("z span", height)
    fdtd.set("material", "Si (Silicon) - Palik")

def simulate_generation_lumerical(fdtd, population, temp_base_path,
                                  geom_lsf, simu_lsf, temp_dir, mode_type="apodized"):
    """
    Simula a geração completa via FDTD usando paralelização runjobs().
    """
    fdtd.clearjobs()
    job_files = []

    # 1. Preparação dos arquivos e adição à fila (Fila de Jobs)
    for i, chromosome in enumerate(population):
        file_name = f"indiv_{i+1}_fdtd.fsp"
        file_path = os.path.join(temp_dir, file_name)
        
        fdtd.load(temp_base_path)
        _build_apodized_structure(fdtd, chromosome)
        
        fdtd.save(file_path)
        fdtd.addjob(file_path)
        job_files.append(file_path)

    # 2. Execução Paralela
    print(f"      [FDTD] Executando simulações paralelas (runjobs)...")
    fdtd.runjobs()

    # 3. Coleta dos Espectros
    all_spectra = [] # Vetores de transmissão que serão passados à fitness
    frequencies = None

    for file_path in job_files:
        try:
            fdtd.load(file_path)
            
            # Coleta o resultado do monitor de transmissão na entrada/saída do guia
            # Como discutido, o FDTD permite capturar o espectro total (T) [cite: 144, 186, 295]
            # Assumimos que o arquivo base possui um monitor chamado "monitor_drop"
            res = fdtd.getresult("monitor_drop", "T")
            
            transmission_vector = abs(res['T'].flatten())
            wavelengths = res['lambda'].flatten()
            current_freqs = 299792458.0 / wavelengths
            
            # Para manter compatibilidade com as fitness functions existentes:
            # Criamos uma matriz fake onde a posição de transmissão (S21) é o nosso espectro
            S_equiv = np.zeros((2, 2, len(current_freqs)), dtype=np.complex128)
            S_equiv[1, 0, :] = np.sqrt(transmission_vector) 
            
            all_spectra.append(S_equiv)
            if frequencies is None:
                frequencies = current_freqs
                
        except Exception as e:
            print(f"      [Erro] Coleta falhou em {os.path.basename(file_path)}: {e}")
            all_spectra.append(None)

    return all_spectra, frequencies