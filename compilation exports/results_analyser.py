import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
# Descobre a pasta raiz do projeto (voltando 1 nível a partir de compilation_exports)
_compiler_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_compiler_dir)

def get_exact_crossing(x1, y1, x2, y2, threshold):
    """Realiza interpolação linear para encontrar o comprimento de onda exato do cruzamento."""
    if y2 == y1: return x1
    return x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)

def calculate_3db_metrics(wl, spectrum_db):
    """Calcula o pico, o centro e a banda 3dB usando interpolação."""
    peak_idx = np.argmax(spectrum_db)
    max_db = spectrum_db[peak_idx]
    threshold_db = max_db - 3.0

    left_idx = peak_idx
    while left_idx > 0 and spectrum_db[left_idx - 1] >= threshold_db:
        left_idx -= 1

    right_idx = peak_idx
    while right_idx < len(spectrum_db) - 1 and spectrum_db[right_idx + 1] >= threshold_db:
        right_idx += 1

    wl_left_exact = wl[left_idx]
    if left_idx > 0:
        wl_left_exact = get_exact_crossing(
            wl[left_idx - 1], spectrum_db[left_idx - 1],
            wl[left_idx], spectrum_db[left_idx],
            threshold_db
        )

    wl_right_exact = wl[right_idx]
    if right_idx < len(spectrum_db) - 1:
        wl_right_exact = get_exact_crossing(
            wl[right_idx], spectrum_db[right_idx],
            wl[right_idx + 1], spectrum_db[right_idx + 1],
            threshold_db
        )

    bw_3db = abs(wl_right_exact - wl_left_exact)
    center_wl_3db = (wl_left_exact + wl_right_exact) / 2.0

    return wl[peak_idx], center_wl_3db, bw_3db, max_db, peak_idx

def calculate_slsr(spectrum_db, peak_idx):
    """
    Isola o lóbulo principal encontrando os primeiros vales (nulls) e 
    calcula o Side Lobe Suppression Ratio (SLSR) nos lóbulos laterais restantes.
    """
    def find_first_null(start_idx, step):
        curr_idx = start_idx
        min_val = spectrum_db[start_idx]
        null_idx = start_idx
        
        while 0 <= curr_idx < len(spectrum_db):
            if spectrum_db[curr_idx] < min_val:
                min_val = spectrum_db[curr_idx]
                null_idx = curr_idx
            # Exige uma subida de 0.5 dB após o mínimo
            elif spectrum_db[curr_idx] > min_val + 0.5:
                break
            curr_idx += step
        return null_idx

    left_null = find_first_null(peak_idx, -1)
    right_null = find_first_null(peak_idx, 1)

    side_lobes = []
    if left_null > 0:
        side_lobes.append(np.max(spectrum_db[:left_null]))
    if right_null < len(spectrum_db) - 1:
        side_lobes.append(np.max(spectrum_db[right_null + 1:]))
        
    if side_lobes:
        max_side_lobe = max(side_lobes)
        slsr = spectrum_db[peak_idx] - max_side_lobe
        return slsr, max_side_lobe
    else:
        margin = spectrum_db[peak_idx] - np.min(spectrum_db)
        return margin, np.min(spectrum_db)

def run_analysis():
    _compiler_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(_compiler_dir, "results_S11_spectra.csv")
    pdf_spectrum_path = os.path.join(_compiler_dir, "best_spectrum_plot.pdf")
    pdf_scatter_path = os.path.join(_compiler_dir, "filter_performance_scatter.pdf")
    json_out_path = os.path.join(_compiler_dir, "spectra_metrics.json")

    if not os.path.exists(csv_path):
        print(f"[ERRO] Arquivo {csv_path} não encontrado!")
        return

    print("==================================================================")
    print(" INICIANDO ANALISADOR DE RESULTADOS (ESPECTROS E DISPERSÃO)")
    print("==================================================================\n")

    print("[1/4] Lendo arquivo de espectros...")
    df = pd.read_csv(csv_path)
    df = df.sort_values(by='Wavelength_nm').reset_index(drop=True)
    wavelengths = df['Wavelength_nm'].values
    
    data_columns = [col for col in df.columns if col != 'Wavelength_nm']
    
    metrics_dict = {}
    best_fitness = -1.0
    best_column = None

    print("[2/4] Calculando métricas (Pico, Banda 3dB e SLSR)...")
    for col in data_columns:
        try:
            fit_val = float(col.split('_')[1])
            if fit_val > best_fitness:
                best_fitness = fit_val
                best_column = col
        except Exception:
            pass

        spectrum_linear = df[col].values
        spectrum_linear = np.clip(spectrum_linear, 1e-12, None)
        spectrum_db = 10 * np.log10(spectrum_linear)

        peak_wl, center_3db, bw_3db, max_db, peak_idx = calculate_3db_metrics(wavelengths, spectrum_db)
        slsr_db, max_side_lobe_db = calculate_slsr(spectrum_db, peak_idx)

        metrics_dict[col] = {
            "fitness": round(fit_val, 4),
            "peak_reflectance_dB": round(float(max_db), 4),
            "peak_wavelength_nm": round(float(peak_wl), 4),
            "center_wavelength_3dB_nm": round(float(center_3db), 4),
            "bandwidth_3dB_nm": round(float(bw_3db), 4),
            "side_lobe_suppression_ratio_dB": round(float(slsr_db), 4),
            "highest_side_lobe_dB": round(float(max_side_lobe_db), 4)
        }

    with open(json_out_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"      -> Métricas salvas em: spectra_metrics.json")

    if best_column:
        print(f"\n[3/4] Plotando o melhor espectro (Fitness: {best_fitness})...")
        best_spectrum_linear = np.clip(df[best_column].values, 1e-12, None)
        best_spectrum_db = 10 * np.log10(best_spectrum_linear)

        plt.figure(figsize=(10, 6))
        
        plt.plot(wavelengths, best_spectrum_db, color='#1f77b4', linewidth=2.5, 
                 label=f'Best Individual (Fitness: {best_fitness:.4f})')
        
        b_metrics = metrics_dict[best_column]
        thresh_3db = b_metrics["peak_reflectance_dB"] - 3.0
        side_lobe_lvl = b_metrics["highest_side_lobe_dB"]
        
        #plt.axhline(thresh_3db, color='red', linestyle='--', alpha=0.7, 
                    #label=f'-3 dB Bandwidth ({b_metrics["bandwidth_3dB_nm"]:.2f} nm)')
        
        plt.axhline(side_lobe_lvl, color='green', linestyle='-.', alpha=0.7, 
                    label=f'SLSR Limit ({b_metrics["side_lobe_suppression_ratio_dB"]:.2f} dB)')
        
        #plt.title('Reflection Spectrum of the Best Optimized Bragg Grating', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Wavelength (nm)', fontsize=16)
        plt.ylabel('Reflectance (dB)', fontsize=16)
        
        plt.grid(True, linestyle=':', alpha=0.8)
        
        plt.ylim(-20, 0)
        plt.yticks(np.linspace(-20,0,21))
        # Zoom no espectro
        plt.xlim(1500, 1550)
        plt.xticks(np.linspace(1500,1550,11))
        plt.tick_params(axis='both', which='major', labelsize=14)

        plt.legend(fontsize=14, loc='lower right')
        plt.tight_layout()

        plt.savefig(pdf_spectrum_path, format='pdf', dpi=300)
        plt.close()
        print(f"      -> Plot salvo em: best_spectrum_plot.pdf")

    # --- NOVO BLOCO: SCATTER PLOT ---
    print("\n[4/4] Plotando o gráfico de dispersão (Performance Global)...")
    
    scatter_x = []
    scatter_y = []
    scatter_c = []

    for col, metrics in metrics_dict.items():
        scatter_x.append(metrics["center_wavelength_3dB_nm"])
        scatter_y.append(metrics["bandwidth_3dB_nm"])
        scatter_c.append(metrics["fitness"])

    if scatter_x:
        plt.figure(figsize=(9, 6))
        
        # Desenha as linhas tracejadas caindo até o eixo X (y=0)
        plt.vlines(x=scatter_x, ymin=0, ymax=scatter_y, color='gray', 
                   linestyle='--', alpha=0.6, linewidth=1.5, zorder=1)
        
        # Plota os pontos usando o colormap 'viridis'
        sc = plt.scatter(scatter_x, scatter_y, c=scatter_c, cmap='viridis', 
                         s=150, edgecolor='black', alpha=0.9, zorder=3)
        
        # Adiciona e estiliza a barra de cores
        cbar = plt.colorbar(sc)
        cbar.set_label('Fitness Score', fontsize=12, fontweight='bold', labelpad=10)
        
        #plt.title('Filter Performance Overview', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Center Wavelength (3dB) [nm]', fontsize=14)
        plt.ylabel('3dB Bandwidth [nm]', fontsize=14)
        
        plt.grid(True, linestyle=':', alpha=0.5, zorder=0)
        
        # --- AJUSTES DE ESCALA ---
        # Força o eixo X a ir de 1440 a 1560 com marcações a cada 10 nm
        plt.xlim(1440, 1560)
        plt.xticks(np.arange(1440, 1561, 10))
        plt.tick_params(axis='both', which='major', labelsize=12)
        # Força o eixo Y a começar no 0 para ancorar as linhas tracejadas perfeitamente
        plt.ylim(bottom=0)
        
        plt.tight_layout()

        plt.savefig(pdf_scatter_path, format='pdf', dpi=300)
        plt.close()
        print(f"      -> Plot salvo em: filter_performance_scatter.pdf")

# --- NOVO BLOCO: ESPECTRO DE MELHOR TRADE-OFF (SLSR / BANDA) ---
    print("\n[5/5] Plotando o espectro com o melhor balanço entre SLSR e Banda...")
    
    best_tradeoff_val = -float('inf')
    best_tradeoff_column = None

    # Encontra a coluna com a melhor razão (SLSR / BW)
    for col, metrics in metrics_dict.items():
        slsr = metrics["side_lobe_suppression_ratio_dB"]
        bw = metrics["bandwidth_3dB_nm"]
        
        # Evita divisão por zero e penaliza bandas negativas ou nulas
        if bw > 0:
            tradeoff_score = slsr / bw
            if tradeoff_score > best_tradeoff_val:
                best_tradeoff_val = tradeoff_score
                best_tradeoff_column = col

    if best_tradeoff_column:
        best_tradeoff_linear = np.clip(df[best_tradeoff_column].values, 1e-12, None)
        best_tradeoff_db = 10 * np.log10(best_tradeoff_linear)
        t_metrics = metrics_dict[best_tradeoff_column]

        plt.figure(figsize=(10, 6))
        
        # Usando a cor roxa (purple) para destacar esse gráfico específico
        plt.plot(wavelengths, best_tradeoff_db, color='#9467bd', linewidth=2.5, 
                 label=f'Best Trade-off (SLSR/BW)\nSLSR: {t_metrics["side_lobe_suppression_ratio_dB"]:.2f} dB | BW: {t_metrics["bandwidth_3dB_nm"]:.2f} nm\n(Fitness: {t_metrics["fitness"]:.4f})')
        
        thresh_3db = t_metrics["peak_reflectance_dB"] - 3.0
        side_lobe_lvl = t_metrics["highest_side_lobe_dB"]
        
        #plt.axhline(thresh_3db, color='red', linestyle='--', alpha=0.7, 
                    #label=f'-3 dB Bandwidth ({t_metrics["bandwidth_3dB_nm"]:.2f} nm)')
        
        plt.axhline(side_lobe_lvl, color='green', linestyle='-.', alpha=0.7, 
                    label=f'Highest Side Lobe ({side_lobe_lvl:.2f} dB)')
        
        #plt.title('Reflection Spectrum with Best Trade-off (SLSR vs. Bandwidth)', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Wavelength (nm)', fontsize=16)
        plt.ylabel('Reflectance (dB)', fontsize=16)
        
        plt.grid(True, linestyle=':', alpha=0.8)
        
        y_min = np.floor(max(-60, np.min(best_tradeoff_db) - 5))
        y_max = 5
        #plt.ylim(y_min, y_max)
        #plt.yticks(np.arange(y_min, y_max + 1, 1))
        plt.ylim(-20, 0)
        plt.yticks(np.linspace(-20,0,21))
        # Zoom no espectro
        plt.xlim(1450, 1490)
        plt.xticks(np.linspace(1450,1490,11))
        # Zoom Dinâmico: Centraliza no pico do filtro com uma janela de 50nm
        center_wl = t_metrics["peak_wavelength_nm"]
        #plt.xlim(center_wl - 25, center_wl + 25)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(fontsize=14, loc='lower right')
        plt.tight_layout()

        pdf_tradeoff_path = os.path.join(_compiler_dir, "best_tradeoff_spectrum_plot.pdf")
        plt.savefig(pdf_tradeoff_path, format='pdf', dpi=300)
        plt.close()
        print(f"      -> Plot salvo em: best_tradeoff_spectrum_plot.pdf")

# --- NOVO BLOCO 6: EVOLUÇÃO DO FITNESS (CONVERGÊNCIA) ---
    print("\n[6/6] Plotando o gráfico de evolução do fitness (Convergência)...")
    
    _project_root = os.path.dirname(_compiler_dir)
    json_files = glob.glob(os.path.join(_project_root, "simulation_results", "*.json"))
    
    plt.figure(figsize=(11, 6))
    
    lines_plotted = 0
    max_generations_plotted = 1
    cmap = plt.get_cmap('tab20')
    
    for file in json_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            if 'fitness_history' in data and data['fitness_history']:
                history = data['fitness_history']
                history_clean = [max(0, val) for val in history]
                best_fitness_run = max(history_clean)
                
                # =========================================================
                # FILTRO ABSOLUTO DE IDENTIDADE
                # Só plota se o fitness do JSON existir na nossa lista oficial (metrics_dict)
                # =========================================================
                matched_wl = None
                for col, metrics in metrics_dict.items():
                    # Compara os valores com 1e-3 de tolerância para evitar erros de arredondamento de float
                    if abs(metrics["fitness"] - best_fitness_run) < 1e-3:
                        matched_wl = metrics["center_wavelength_3dB_nm"]
                        break
                
                # Se o JSON não faz parte dos "aprovados", é um fantasma/falha. Ignora!
                if matched_wl is None:
                    continue
                
                # Dupla segurança física: Ignora anomalias abaixo da banda de telecom
                if matched_wl < 1400.0:
                    continue
                
                label_str = f"{matched_wl:.2f} nm"
                
                generations = range(1, len(history_clean) + 1)
                
                if len(history_clean) > max_generations_plotted:
                    max_generations_plotted = len(history_clean)
                    
                color = cmap(lines_plotted % 20)
                
                plt.plot(generations, history_clean, linewidth=2, alpha=0.85, 
                         color=color, label=label_str)
                lines_plotted += 1
                    
        except Exception as e:
            pass

    if lines_plotted > 0:
        #plt.title('Genetic Algorithm Convergence History', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Generation', fontsize=16)
        plt.ylabel('Best Fitness Score', fontsize=16)
        
        plt.grid(True, linestyle=':', alpha=0.7)
        
        plt.ylim(0.4, 1.0)
        plt.xlim(1, max_generations_plotted)
        plt.tick_params(axis='both', which='major', labelsize=14)
        handles, labels = plt.gca().get_legend_handles_labels()
        try:
            hl = sorted(zip(handles, labels), key=lambda x: float(x[1].split()[0]))
            handles2, labels2 = zip(*hl)
            # Legenda fixada exatamente ao lado direito do eixo (x=1.02) no centro (y=0.5)
            plt.legend(handles2, labels2, fontsize=14, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Center WL (3dB)", title_fontsize=15)
        except Exception:
            plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Center WL", title_fontsize=15)

        # Removemos o rect=[] para o gráfico parar de se espremer
        plt.tight_layout() 
        
        pdf_convergence_path = os.path.join(_compiler_dir, "fitness_convergence_plot.pdf")
        
        # O segredo está no bbox_inches='tight': ele expande o PDF para abraçar a legenda externa
        plt.savefig(pdf_convergence_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      -> Plot salvo em: fitness_convergence_plot.pdf")
    else:
        print("      -> [AVISO] Nenhum dado válido encontrado para plotagem.")

# --- NOVO BLOCO 7: ESPECTRO DE MENOR BANDA (NARROWEST BANDWIDTH) ---
    print("\n[7/7] Plotando o espectro com a menor banda 3dB...")
    
    min_bw_val = float('inf')
    min_bw_column = None

    # Encontra a coluna com a menor banda 3dB válida
    for col, metrics in metrics_dict.items():
        bw = metrics["bandwidth_3dB_nm"]
        if 0 < bw < min_bw_val:
            min_bw_val = bw
            min_bw_column = col

    if min_bw_column:
        min_bw_linear = np.clip(df[min_bw_column].values, 1e-12, None)
        min_bw_db = 10 * np.log10(min_bw_linear)
        nb_metrics = metrics_dict[min_bw_column]

        plt.figure(figsize=(10, 6))
        
        # Usando a cor ciano/teal para diferenciar dos demais gráficos
        plt.plot(wavelengths, min_bw_db, color='#17becf', linewidth=2.5, 
                 label=f'Narrowest Bandwidth: {nb_metrics["bandwidth_3dB_nm"]:.2f} nm\nSLSR: {nb_metrics["side_lobe_suppression_ratio_dB"]:.2f} dB\n(Fitness: {nb_metrics["fitness"]:.4f})')
        
        thresh_3db = nb_metrics["peak_reflectance_dB"] - 3.0
        side_lobe_lvl = nb_metrics["highest_side_lobe_dB"]
        
        #plt.axhline(thresh_3db, color='red', linestyle='--', alpha=0.7, 
                    #label=f'-3 dB Level ({thresh_3db:.2f} dB)')
        
        plt.axhline(side_lobe_lvl, color='green', linestyle='-.', alpha=0.7, 
                    label=f'Highest Side Lobe ({side_lobe_lvl:.2f} dB)')
        
        #plt.title('Reflection Spectrum with Narrowest 3dB Bandwidth', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Wavelength (nm)', fontsize=16)
        plt.ylabel('Reflectance (dB)', fontsize=16)
        
        plt.grid(True, linestyle=':', alpha=0.8)
        
        y_min = np.floor(max(-60, np.min(min_bw_db) - 5))
        plt.ylim(-20, 0)
        # Zoom no espectro
        plt.xlim(1430, 1470)
        plt.yticks(np.linspace(-20,0,21))
        plt.xticks(np.linspace(1430,1470,11))
        plt.tick_params(axis='both', which='major', labelsize=14)
        # Zoom Dinâmico: Centraliza no pico do filtro com uma janela de 50nm
        center_wl = nb_metrics["peak_wavelength_nm"]
        #plt.xlim(center_wl - 25, center_wl + 25)
        
        plt.legend(fontsize=14, loc='lower right')
        plt.tight_layout()

        pdf_nb_path = os.path.join(_compiler_dir, "narrowest_bandwidth_spectrum_plot.pdf")
        plt.savefig(pdf_nb_path, format='pdf', dpi=300)
        plt.close()
        print(f"      -> Plot salvo em: narrowest_bandwidth_spectrum_plot.pdf")
    else:
        print("      -> [AVISO] Não foi possível encontrar um espectro com banda válida.")


if __name__ == "__main__":
    run_analysis()