import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# --- CONFIGURAÇÃO ---
RESULTS_FOLDER = "simulation_results" 
EXPORT_FOLDER = "compilation exports"  # Nova pasta de destino
FITNESS_THRESHOLD = 0.75

def compile_and_plot_results(folder_path):
    print(f"Buscando arquivos JSON em: {folder_path}...")
    
    # --- MUDANÇA: Cria a pasta de exportação se não existir ---
    if not os.path.exists(EXPORT_FOLDER):
        os.makedirs(EXPORT_FOLDER)
        print(f"Pasta '{EXPORT_FOLDER}' criada com sucesso.")
    
    scatter_data = []
    evolution_data = []
    
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print("Nenhum arquivo JSON encontrado.")
        return

    count_processed = 0
    count_filtered = 0

    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                fitness = data.get("best_fitness_so_far", 0.0)
                
                if fitness > FITNESS_THRESHOLD:
                    filename = os.path.basename(filepath)
                    analysis = data.get("analysis_of_best_in_gen", {})
                    peak_wl = analysis.get("real_peak_wl_nm")
                    bw = analysis.get("real_bw_nm")
                    
                    if peak_wl is not None and bw is not None:
                        # 1. Dados Scatter
                        scatter_data.append({
                            "Filename": filename,
                            "Fitness": fitness,
                            "Peak_Wavelength_nm": peak_wl,
                            "Bandwidth_nm": bw
                        })
                    
                        # 2. Dados Evolução
                        history = data.get("fitness_history", [])
                        if history:
                            evolution_data.append({
                                "Filename": filename,
                                "History": history,
                                "FinalFitness": fitness,
                                "PeakWL": peak_wl 
                            })
                        
                        count_filtered += 1
                count_processed += 1
        except Exception as e:
            print(f"Erro ao ler {os.path.basename(filepath)}: {e}")

    print(f"Processados: {count_processed}. Filtrados (> {FITNESS_THRESHOLD}): {count_filtered}.")

    if not scatter_data:
        print("Nenhum dado atendeu aos critérios.")
        return

    # ==========================================
    # GRÁFICO 1: Scatter Plot
    # ==========================================
    df = pd.DataFrame(scatter_data)
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    ax1.vlines(
        x=df["Peak_Wavelength_nm"], ymin=0, ymax=df["Bandwidth_nm"], 
        colors='black', linestyles='dashed', alpha=0.8, linewidth=1.5
    )
    scatter = ax1.scatter(
        df["Peak_Wavelength_nm"], df["Bandwidth_nm"], 
        c=df["Fitness"], cmap='viridis', s=110, alpha=1.0, edgecolors='k', zorder=3
    )
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Fitness Score', fontsize=11, weight='bold')
    
    # Título removido para LaTeX
    # ax1.set_title(...) 
    
    ax1.set_xlabel('Center Wavelength(nm)', fontsize=12, weight='bold')
    ax1.set_ylabel('Bandwidth(nm)', fontsize=12, weight='bold')
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    
    # --- MUDANÇA: Salvar dentro da pasta EXPORT_FOLDER ---
    output_path1 = os.path.join(EXPORT_FOLDER, "compiled_results_scatter.pdf")
    plt.savefig(output_path1, format='pdf', bbox_inches='tight')
    print(f"Gráfico 1 salvo em: {output_path1}")

    # ==========================================
    # GRÁFICO 2: Evolução
    # ==========================================
    if evolution_data:
        fig2, ax2 = plt.subplots(figsize=(13, 8))
        
        evolution_data.sort(key=lambda x: x["PeakWL"])
        
        all_wls = [item["PeakWL"] for item in evolution_data]
        norm = plt.Normalize(min(all_wls), max(all_wls))
        cmap = plt.cm.turbo 
        
        for item in evolution_data:
            history = item["History"]
            generations = range(1, len(history) + 1)
            peak_wl = item["PeakWL"]
            final_fit = item["FinalFitness"]
            
            color = cmap(norm(peak_wl))
            label_text = f"{peak_wl:.1f} nm"
            
            ax2.plot(generations, history, label=label_text, color=color, linewidth=2, alpha=0.8)
            ax2.scatter(len(history), final_fit, color=color, edgecolors='k', s=50, zorder=4)

        # Título removido para LaTeX
        # ax2.set_title(...)
        
        ax2.set_xlabel('Geração', fontsize=12, weight='bold')
        ax2.set_ylabel('Melhor Aptidão', fontsize=12, weight='bold')
        
        ax2.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.6)
        ax2.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
        ax2.minorticks_on()
        
        ax2.legend(title="Centro (nm)", title_fontsize='11', fontsize='10', 
                   loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        plt.tight_layout()
        
        # --- MUDANÇA: Salvar dentro da pasta EXPORT_FOLDER ---
        output_path2 = os.path.join(EXPORT_FOLDER, "compiled_fitness_evolution_legend.pdf")
        plt.savefig(output_path2, format='pdf', bbox_inches='tight')
        print(f"Gráfico 2 salvo em: {output_path2}")
        
    plt.show()

if __name__ == "__main__":
    if os.path.exists(RESULTS_FOLDER):
        compile_and_plot_results(RESULTS_FOLDER)
    else:
        print(f"Pasta '{RESULTS_FOLDER}' não encontrada. Tentando diretório atual...")
        compile_and_plot_results(".")