# utils/parameter_extraction.py

import numpy as np

# Constantes globais para o cálculo
GUIDE_LENGTH = 40e-6  # O comprimento total do guia de onda final
C_LIGHT = 299792458.0 # Velocidade da luz no vácuo (m/s)

def _extract_effective_parameters(S_raw, S_ref, period_length, frequencies):
    """
    Extrai o índice de refração (n) e a impedância (z) efetivos de uma célula unitária.
    """
    # Extrai os parâmetros S relevantes (S11 e S21)
    S11_raw = S_raw[0, 0, :]
    S21_raw = S_raw[1, 0, :]
    S21_ref = S_ref[1, 0, :]
    
    # "Des-incorporação" (de-embedding) para isolar o efeito da perturbação
    S21_deembedded = S21_raw / S21_ref
    S11_deembedded = S11_raw
    
    # --- Cálculo de n e z segundo a documentação da Ansys ---
    z_eff_sq = ((1 + S11_deembedded)**2 - S21_deembedded**2) / ((1 - S11_deembedded)**2 - S21_deembedded**2)
    z_eff = np.sqrt(z_eff_sq)
    z_eff[np.real(z_eff) < 0] *= -1

    term_for_acos = (1 - S11_deembedded**2 + S21_deembedded**2) / (2 * S21_deembedded)
    term_for_acos = np.clip(term_for_acos, -1, 1)
    
    kd = np.arccos(term_for_acos)
    k0 = 2 * np.pi * frequencies / C_LIGHT
    
    n_eff = (1 / (k0 * period_length)) * kd
    n_eff[np.imag(n_eff) < 0] *= -1
    
    return n_eff, z_eff

def _calculate_s_matrix_from_params(n_eff, z_eff, frequencies):
    """
    Calcula a Matriz S COMPLETA de uma placa de material com parâmetros efetivos n e z
    e comprimento total GUIDE_LENGTH.
    """
    k0 = 2 * np.pi * frequencies / C_LIGHT
    
    # Coeficiente de reflexão na interface
    gamma = (z_eff - 1) / (z_eff + 1)
    
    # Termo de propagação através do comprimento total do guia
    propagation_term = np.exp(1j * n_eff * k0 * GUIDE_LENGTH)
    
    # Fórmulas de Fabry-Pérot para uma placa (slab)
    S11 = gamma * (1 - propagation_term**2) / (1 - gamma**2 * propagation_term**2)
    S21 = propagation_term * (1 - gamma**2) / (1 - gamma**2 * propagation_term**2)
    
    # Monta a matriz S completa (assumindo reciprocidade: S12=S21, S22=S11)
    num_freq = len(frequencies)
    S_total = np.zeros((2, 2, num_freq), dtype=np.complex128)
    S_total[0, 0, :] = S11
    S_total[1, 0, :] = S21
    S_total[0, 1, :] = S21 # S12 = S21
    S_total[1, 1, :] = S11 # S22 = S11
    
    return S_total

def post_process_generation(
    S_matrixes_for_generation,
    S_matrixes_ref_for_generation,
    current_population,
    frequencies
):
    """
    Função principal que executa o fluxo de extração para uma geração inteira.
    Retorna as matrizes S totais e fisicamente corretas.
    """
    if frequencies is not None:
        frequencies = frequencies.flatten()
    all_total_S_matrices = []
    
    if not all([S_matrixes_for_generation, S_matrixes_ref_for_generation, current_population]):
        print("AVISO: Uma das listas de entrada para post_process_generation está vazia.")
        return np.array([])

    for S_raw, S_ref, chromosome in zip(S_matrixes_for_generation, S_matrixes_ref_for_generation, current_population):
        if S_raw is None or S_ref is None:
            # Adiciona uma matriz de zeros como placeholder para simulações que falharam
            num_freq = len(frequencies)
            all_total_S_matrices.append(np.zeros((2, 2, num_freq), dtype=np.complex128))
            continue

        Lambda = chromosome['Lambda']
        
        try:
            # 1. Extrai os parâmetros efetivos (n, z)
            n_eff, z_eff = _extract_effective_parameters(S_raw, S_ref, Lambda, frequencies)
            
            # 2. Calcula a matriz S total a partir dos parâmetros
            S_total = _calculate_s_matrix_from_params(n_eff, z_eff, frequencies)
            
            all_total_S_matrices.append(S_total)

        except Exception as e:
            print(f"!!! Erro durante o pós-processamento de um cromossomo: {e}")
            num_freq = len(frequencies)
            all_total_S_matrices.append(np.zeros((2, 2, num_freq), dtype=np.complex128))

    # Empilha a lista de matrizes 3D para formar o array 4D final
    return np.stack(all_total_S_matrices, axis=0)