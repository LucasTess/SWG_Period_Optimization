import os
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Constante da velocidade da luz no vácuo (m/s)
C_LIGHT = 299792458.0

def generate_gaussian_spectrum(frequencies, center_frequency, fwhm, max_amplitude=3.0):
    """
    Gera um espectro de pulso gaussiano (sinal de entrada).
    """
    if fwhm <= 0:
        # Se a largura de banda for zero ou negativa, retorna um array de zeros
        return np.zeros_like(frequencies)
        
    # Converte FWHM para desvio padrão (sigma)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    exponent = -((frequencies - center_frequency)**2) / (2 * sigma**2)
    return max_amplitude * np.exp(exponent)

def _calculate_single_delta_amp(reflected_spectrum):
    """
    Calcula o fitness 'delta_amp' para um único espectro de amplitude.
    """
    # Encontra picos e vales. Os parâmetros podem ser ajustados se necessário.
    peak_indices, _ = find_peaks(reflected_spectrum, height=0.05, distance=5)
    valley_indices, _ = find_peaks(-reflected_spectrum, distance=5)
    
    # Requer ao menos um pico e um vale para calcular a variação
    if len(peak_indices) < 1 or len(valley_indices) < 1:
        return 0.0

    extrema_indices = np.sort(np.concatenate([peak_indices, valley_indices]))
    extrema_amplitudes = reflected_spectrum[extrema_indices]
    
    # Retorna a soma das amplitudes entre picos e vales consecutivos
    return np.sum(np.abs(np.diff(extrema_amplitudes)))

def calculate_fitness_for_generation(
    total_S_matrices, 
    frequencies, 
    gaussian_center_freq,
    gaussian_fwhm,
    plot_best_spectrum=False, 
    generation_num=None,
    output_directory="."
):
    """
    Recebe as matrizes S totais e calcula o fitness delta_amp para cada cromossomo,
    usando um pulso de entrada gaussiano com parâmetros definidos.
    """
    # Garante que os arrays tenham o formato correto
    frequencies = frequencies.flatten()
    
    if total_S_matrices.ndim != 4 or total_S_matrices.shape[1:3] != (2, 2):
        raise ValueError("O array de entrada 'total_S_matrices' deve ter o formato [N, 2, 2, F].")
        
    # 1. Gera o espectro de entrada gaussiano com os parâmetros explícitos
    input_spectrum = generate_gaussian_spectrum(
        frequencies,
        gaussian_center_freq,
        gaussian_fwhm
    )
    
    # 2. Extrai os espectros de S11 (complexos) e calcula a magnitude
    s11_complex_spectrums = total_S_matrices[:, 0, 0, :]
    s11_magnitude_spectrums = np.abs(s11_complex_spectrums)
    
    # 3. Calcula os espectros do sinal refletido
    reflected_spectrums = s11_magnitude_spectrums * input_spectrum
    
    # 4. Calcula o fitness (delta_amp) para cada cromossomo
    fitness_values = np.apply_along_axis(_calculate_single_delta_amp, axis=1, arr=reflected_spectrums)
    
    # 5. Opcional: Plota o diagnóstico do melhor cromossomo da geração
    if plot_best_spectrum and generation_num is not None:
        try:
            best_chrom_index = np.argmax(fitness_values)
            best_s11_spectrum = s11_magnitude_spectrums[best_chrom_index]
            best_reflected_spectrum = reflected_spectrums[best_chrom_index]
            best_fitness = fitness_values[best_chrom_index]
            wavelengths_nm = (C_LIGHT / frequencies) * 1e9

            # Gera e salva os dois gráficos de diagnóstico
            _save_diagnostic_plots(wavelengths_nm, input_spectrum, best_s11_spectrum, 
                                   best_reflected_spectrum, best_fitness, 
                                   generation_num, output_directory)
        except Exception as e:
            print(f"  [Debug Plot] !!! Erro ao gerar gráficos de diagnóstico: {e}")
            
    return fitness_values

def _save_diagnostic_plots(wavelengths_nm, input_spectrum, s11_spectrum, 
                           reflected_spectrum, fitness, gen_num, output_dir):
    """
    Função auxiliar para gerar e salvar os dois gráficos de diagnóstico.
    """
    # --- GRÁFICO 1: Resposta Intrínseca |S11| ---
    plt.figure(figsize=(12, 7))
    plt.plot(wavelengths_nm, s11_spectrum, color='purple', linewidth=2, label='Reflexão |S11| (Melhor Cromossomo)')
    plt.title(f'Resposta Intrínseca |S11| - Melhor Cromossomo (Geração {gen_num})', fontsize=16)
    plt.xlabel("Comprimento de Onda (nm)", fontsize=12)
    plt.ylabel("Magnitude |S11|", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend(); plt.ylim(0, 1.1)
    plt.xlim(wavelengths_nm.min(), wavelengths_nm.max())
    s11_plot_path = os.path.join(output_dir, "realtime_S11_spectrum.png")
    plt.savefig(s11_plot_path, dpi=150)
    plt.close()

    # --- GRÁFICO 2: Espectro Refletido Final ---
    peaks, _ = find_peaks(reflected_spectrum, height=0.05, distance=5)
    valleys, _ = find_peaks(-reflected_spectrum, distance=5)
    plt.figure(figsize=(12, 7))
    plt.plot(wavelengths_nm, input_spectrum, 'k--', alpha=0.6, label='Pulso de Entrada (Gaussiano)')
    plt.plot(wavelengths_nm, reflected_spectrum, color='blue', linewidth=2, label='Espectro Refletido Final')
    plt.plot(wavelengths_nm[peaks], reflected_spectrum[peaks], "x", color='red', markersize=8, mew=2, label='Picos Detectados')
    plt.plot(wavelengths_nm[valleys], reflected_spectrum[valleys], "o", color='green', markersize=7, label='Vales Detectados')
    title = f"Espectro Refletido Final - Melhor Cromossomo (Geração {gen_num})\nFitness (delta_amp) = {fitness:.4f}"
    plt.title(title, fontsize=16)
    plt.xlabel("Comprimento de Onda (nm)", fontsize=12)
    plt.ylabel("Amplitude Refletida", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend(); plt.ylim(bottom=0)
    plt.xlim(wavelengths_nm.min(), wavelengths_nm.max())
    reflected_plot_path = os.path.join(output_dir, "realtime_reflected_spectrum.png")
    plt.savefig(reflected_plot_path, dpi=150)
    plt.close()
    
    print(f"  [Debug Plots] Gráficos de diagnóstico salvos em: {output_dir}")