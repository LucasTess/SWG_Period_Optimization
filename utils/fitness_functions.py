# utils/fitness_functions.py (Modificado para aceitar arrays NumPy)

import numpy as np
from abc import ABC, abstractmethod

# Classe base com a nova assinatura
class FitnessStrategy(ABC):
    @abstractmethod
    def calculate(self, S_matrix_total: np.ndarray, frequencies: np.ndarray) -> float:
        """
        Calcula o fitness diretamente de arrays NumPy (Matriz S e frequências).
        """
        pass

class DeltaAmpStrategy(FitnessStrategy):
    """
    Calcula o fitness 'delta_amp' com base na soma das diferenças entre
    picos e vales do espectro de REFLEXÃO (|S11|).
    """
    def calculate(self, S_matrix_total: np.ndarray, frequencies: np.ndarray) -> float:
        try:
            # Pega a amplitude do espectro de REFLEXÃO (|S11|)
            spectrum_amplitude = np.abs(S_matrix_total[0, 0, :]).flatten()
            
            if frequencies.size == 0 or spectrum_amplitude.size == 0:
                return -np.inf
                
            peaks, valleys = [], []
            for i in range(1, len(spectrum_amplitude) - 1):
                if spectrum_amplitude[i] > spectrum_amplitude[i-1] and spectrum_amplitude[i] > spectrum_amplitude[i+1]:
                    peaks.append((i, spectrum_amplitude[i]))
                elif spectrum_amplitude[i] < spectrum_amplitude[i-1] and spectrum_amplitude[i] < spectrum_amplitude[i+1]:
                    valleys.append((i, spectrum_amplitude[i]))
            
            if not peaks or not valleys: 
                return 0.0
                
            total_delta_amp = 0.0
            for peak_idx, peak_val in peaks:
                next_valley_val = None
                for valley_idx, valley_val in valleys:
                    if valley_idx > peak_idx:
                        next_valley_val = valley_val
                        break
                if next_valley_val is not None:
                    total_delta_amp += abs(peak_val - next_valley_val)
                    
            return total_delta_amp
        except Exception:
            return -np.inf

class BandpassStrategy(FitnessStrategy):
    """
    Estratégia para filtro passa-banda, baseada na potência de
    TRANSMISSÃO (|S21|^2).
    """
    def __init__(self, f_center: float, bandwidth: float, transition_bandwidth: float,
                 w_rejection: float, w_passband: float, w_transition: float):
        self.f_center = f_center
        self.bandwidth = bandwidth
        self.transition_bandwidth = transition_bandwidth
        self.w_rej = w_rejection
        self.w_pass = w_passband
        self.w_trans = w_transition
        
    def calculate(self, S_matrix_total: np.ndarray, frequencies: np.ndarray) -> float:
        try:
            # Pega a potência do espectro de TRANSMISSÃO (|S21|^2)
            transmission = np.abs(S_matrix_total[1, 0, :])**2
            
            f_pass_min = self.f_center - (self.bandwidth / 2)
            f_pass_max = self.f_center + (self.bandwidth / 2)
            f_trans1_min = f_pass_min - self.transition_bandwidth
            f_trans2_max = f_pass_max + self.transition_bandwidth
            
            pass_band_mask = (frequencies >= f_pass_min) & (frequencies <= f_pass_max)
            transition1_mask = (frequencies >= f_trans1_min) & (frequencies < f_pass_min)
            transition2_mask = (frequencies > f_pass_max) & (frequencies <= f_trans2_max)
            non_rejection_mask = (frequencies >= f_trans1_min) & (frequencies <= f_trans2_max)
            stop_band_mask = ~non_rejection_mask
            
            # --- Cálculo dos Scores ---
            T_stop_all = transmission[stop_band_mask]
            score_rej = 1.0 - np.max(T_stop_all) if T_stop_all.size > 0 else 0.0

            T_pass = transmission[pass_band_mask]
            if T_pass.size == 0:
                score_pass = 0.0
            else:
                mean_pass = np.mean(T_pass)
                std_dev_pass = np.std(T_pass)
                score_pass = max(0, mean_pass - std_dev_pass)

            T_trans1 = transmission[transition1_mask]
            score_trans1 = max(0, T_trans1[-1] - T_trans1[0]) if T_trans1.size >= 2 else 0.0 # Subida
            
            T_trans2 = transmission[transition2_mask]
            score_trans2 = max(0, T_trans2[0] - T_trans2[-1]) if T_trans2.size >= 2 else 0.0 # Queda
            
            score_trans = min(score_trans1, score_trans2)

            final_fitness = (self.w_rej * score_rej + 
                             self.w_pass * score_pass + 
                             self.w_trans * score_trans)
            
            return float(final_fitness) if not np.isnan(final_fitness) else -np.inf
        except Exception as e:
            print(f"ERRO ao calcular o fitness Bandpass: {e}")
            return -np.inf

class LowpassStrategy(FitnessStrategy):
    """
    Estratégia para filtro passa-baixas, baseada na potência de
    TRANSMISSÃO (|S21|^2).
    """
    def __init__(self, f_cutoff: float, transition_bandwidth: float, 
                 w_rejection: float, w_passband: float, w_transition: float):
        self.f_cutoff = f_cutoff
        self.transition_bandwidth = transition_bandwidth
        self.w_rej = w_rejection
        self.w_pass = w_passband
        self.w_trans = w_transition
    
    def calculate(self, S_matrix_total: np.ndarray, frequencies: np.ndarray) -> float:
        try:
            transmission = np.abs(S_matrix_total[1, 0, :])**2

            f_min_transition = self.f_cutoff - (self.transition_bandwidth / 2)
            f_max_transition = self.f_cutoff + (self.transition_bandwidth / 2)

            pass_band_mask = frequencies < f_min_transition
            stop_band_mask = frequencies > f_max_transition
            transition_mask = (frequencies >= f_min_transition) & (frequencies <= f_max_transition)

            T_stop = transmission[stop_band_mask]
            score_rej = 1.0 - np.max(T_stop) if T_stop.size > 0 else 0.0

            T_pass = transmission[pass_band_mask]
            if T_pass.size == 0:
                score_pass = 0.0
            else:
                mean_pass = np.mean(T_pass)
                std_dev_pass = np.std(T_pass)
                score_pass = max(0, mean_pass - std_dev_pass)

            T_transition = transmission[transition_mask]
            if T_transition.size < 2:
                score_trans = 0.0
            else:
                score_trans = max(0, T_transition[0] - T_transition[-1]) # Queda

            final_fitness = (self.w_rej * score_rej + 
                             self.w_pass * score_pass + 
                             self.w_trans * score_trans)
            
            return float(final_fitness) if not np.isnan(final_fitness) else -np.inf
        except Exception as e:
            print(f"ERRO ao calcular o fitness Lowpass: {e}")
            return -np.inf

class HighpassStrategy(FitnessStrategy):
    """
    Estratégia para filtro passa-altas, baseada na potência de
    TRANSMISSÃO (|S21|^2).
    """
    def __init__(self, f_cutoff: float, transition_bandwidth: float, 
                 w_rejection: float, w_passband: float, w_transition: float):
        self.f_cutoff = f_cutoff
        self.transition_bandwidth = transition_bandwidth
        self.w_rej = w_rejection
        self.w_pass = w_passband
        self.w_trans = w_transition
    
    def calculate(self, S_matrix_total: np.ndarray, frequencies: np.ndarray) -> float:
        try:
            transmission = np.abs(S_matrix_total[1, 0, :])**2

            f_min_transition = self.f_cutoff - (self.transition_bandwidth / 2)
            f_max_transition = self.f_cutoff + (self.transition_bandwidth / 2)

            stop_band_mask = frequencies < f_min_transition
            pass_band_mask = frequencies > f_max_transition
            transition_mask = (frequencies >= f_min_transition) & (frequencies <= f_max_transition)

            T_stop = transmission[stop_band_mask]
            score_rej = 1.0 - np.max(T_stop) if T_stop.size > 0 else 0.0

            T_pass = transmission[pass_band_mask]
            if T_pass.size == 0:
                score_pass = 0.0
            else:
                mean_pass = np.mean(T_pass)
                std_dev_pass = np.std(T_pass)
                score_pass = max(0, mean_pass - std_dev_pass)

            T_transition = transmission[transition_mask]
            if T_transition.size < 2:
                score_trans = 0.0
            else:
                score_trans = max(0, T_transition[-1] - T_transition[0]) # Subida

            final_fitness = (self.w_rej * score_rej + 
                             self.w_pass * score_pass + 
                             self.w_trans * score_trans)
            
            return float(final_fitness) if not np.isnan(final_fitness) else -np.inf
        except Exception as e:
            print(f"ERRO ao calcular o fitness Highpass: {e}")
            return -np.inf