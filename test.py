import numpy as np
import matplotlib.pyplot as plt

# --- 1. Equação Híbrida Universal (10 Parâmetros) ---
def generate_profile_hybrid(N, ds_max, W0, P, M, W1, Q, W2, K, W3, S):
    n = np.arange(1, N + 1)
    x = (2.0 * n - N) / N
    y_raw = W0 * (np.maximum(0, 1.0 - np.abs(x)**P))**M + W1 * (np.abs(x)**Q) + W2 * np.cos(K * np.pi * x) + W3 * x
    
    y_min, y_max = np.min(y_raw), np.max(y_raw)
    y_norm = (y_raw - y_min) / (y_max - y_min) if y_max > y_min else np.ones_like(y_raw)
    if S > 1: y_norm = np.round(y_norm * (S - 1)) / (S - 1)
    return x, y_norm * ds_max

# --- 2. Gerador de Fourier Espectral (4 Parâmetros) ---
def generate_profile_fourier(N, ds_max, H, alpha, beta, S):
    n = np.arange(1, N + 1)
    x = (2.0 * n - N) / N
    y_raw = np.zeros_like(x)
    for k in range(1, int(H) + 1):
        Ak = np.sin(k * beta) / (k ** alpha)
        y_raw += Ak * np.cos(k * np.pi * x / 2.0)
        
    y_min, y_max = np.min(y_raw), np.max(y_raw)
    y_norm = (y_raw - y_min) / (y_max - y_min) if y_max > y_min else np.ones_like(y_raw)
    if S > 1: y_norm = np.round(y_norm * (S - 1)) / (S - 1)
    return x, y_norm * ds_max

# --- 3. Gerador de Fourier Esparsa (5 Parâmetros) ---
def generate_profile_sparse_fourier(N, ds_max, W1, k1, W2, k2, S):
    n = np.arange(1, N + 1)
    x = (2.0 * n - N) / N
    term1 = W1 * np.cos(k1 * np.pi * x / 2.0)
    term2 = W2 * np.cos(k2 * np.pi * x / 2.0)
    y_raw = term1 + term2
    
    y_min, y_max = np.min(y_raw), np.max(y_raw)
    y_norm = (y_raw - y_min) / (y_max - y_min) if y_max > y_min else np.ones_like(y_raw)
    if S > 1: y_norm = np.round(y_norm * (S - 1)) / (S - 1)
    return x, y_norm * ds_max

# --- Configurações da Simulação Visual ---
N_points = 450
ds_max_val = 100

comparisons = [
    {
        "title": "1. Sino Clássico",
        "hybrid": {"W0":1, "P":2, "M":2, "W1":0, "Q":1, "W2":0, "K":1, "W3":0, "S":0},
        "fourier_4": {"H":1, "alpha":1.0, "beta":1.57, "S":0},
        "sparse_5": {"W1":1, "k1":1, "W2":0, "k2":1, "S":0} 
    },
    {
        "title": "2. Platô Largo (Flat-Top)",
        "hybrid": {"W0":1, "P":6, "M":1, "W1":0, "Q":1, "W2":0, "K":1, "W3":0, "S":0},
        "fourier_4": {"H":7, "alpha":1.5, "beta":1.57, "S":0},
        "sparse_5": {"W1":1, "k1":1, "W2":-0.15, "k2":3, "S":0} 
    },
    {
        "title": "3. Perfil Invertido (U)",
        "hybrid": {"W0":0, "P":1, "M":1, "W1":1, "Q":2, "W2":0, "K":1, "W3":0, "S":0},
        "fourier_4": {"H":1, "alpha":1.0, "beta":-1.57, "S":0},
        "sparse_5": {"W1":-1, "k1":1, "W2":0, "k2":1, "S":0} 
    },
    {
        "title": "4. Sino com Ripples",
        "hybrid": {"W0":1, "P":2, "M":2, "W1":0, "Q":1, "W2":0.3, "K":8, "W3":0, "S":0},
        "fourier_4": {"H":6, "alpha":0.4, "beta":2.5, "S":0},
        "sparse_5": {"W1":1, "k1":1, "W2":0.2, "k2":12, "S":0} 
    },
    {
        "title": "5. Escada Simétrica",
        "hybrid": {"W0":1, "P":2, "M":1, "W1":0, "Q":1, "W2":0, "K":1, "W3":0, "S":6},
        "fourier_4": {"H":1, "alpha":1.0, "beta":1.57, "S":6},
        "sparse_5": {"W1":1, "k1":1, "W2":0, "k2":1, "S":6}
    }
]

# --- Plotagem ---
fig, axes = plt.subplots(5, 3, figsize=(18, 16))
fig.suptitle("Showdown: Híbrida (10 Genes) vs Fourier Espectral (4) vs Fourier Esparsa (5)", fontsize=18, y=0.98)

for i, comp in enumerate(comparisons):
    # Coluna 0: Híbrida
    x, ds_hyb = generate_profile_hybrid(N_points, ds_max_val, **comp["hybrid"])
    axes[i, 0].plot(x, ds_hyb, color='darkblue', linewidth=2)
    axes[i, 0].fill_between(x, ds_hyb, alpha=0.2, color='darkblue')
    axes[i, 0].set_title(f"Híbrida: {comp['title']}", fontsize=11)
    axes[i, 0].grid(True, linestyle='--', alpha=0.6)
    axes[i, 0].set_ylim(-5, ds_max_val + 10)
    
    # Coluna 1: Fourier 4 Genes
    x, ds_fou4 = generate_profile_fourier(N_points, ds_max_val, **comp["fourier_4"])
    axes[i, 1].plot(x, ds_fou4, color='purple', linewidth=2)
    axes[i, 1].fill_between(x, ds_fou4, alpha=0.2, color='purple')
    axes[i, 1].set_title(f"Fourier Espectral (4g): {comp['title']}", fontsize=11)
    axes[i, 1].grid(True, linestyle='--', alpha=0.6)
    axes[i, 1].set_ylim(-5, ds_max_val + 10)

    # Coluna 2: Fourier Esparsa 5 Genes
    x, ds_fou5 = generate_profile_sparse_fourier(N_points, ds_max_val, **comp["sparse_5"])
    axes[i, 2].plot(x, ds_fou5, color='darkred', linewidth=2)
    axes[i, 2].fill_between(x, ds_fou5, alpha=0.2, color='darkred')
    axes[i, 2].set_title(f"Fourier Esparsa (5g): {comp['title']}", fontsize=11)
    axes[i, 2].grid(True, linestyle='--', alpha=0.6)
    axes[i, 2].set_ylim(-5, ds_max_val + 10)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()