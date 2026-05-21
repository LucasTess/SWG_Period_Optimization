import os
import glob
import json
import re

def cleanup_failed_experiments():
    # --- Mapeamento de Caminhos Inteligente ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.basename(current_dir)
    
    # Se o script já está dentro de simulation_results, a pasta alvo é a atual
    if folder_name == "simulation_results":
        sim_results_dir = current_dir
    elif folder_name == "compilation_exports":
        root_dir = os.path.dirname(current_dir)
        sim_results_dir = os.path.join(root_dir, "simulation_results")
    else:
        # Se estiver na raiz do projeto
        sim_results_dir = os.path.join(current_dir, "simulation_results")
    
    if not os.path.exists(sim_results_dir):
        print(f"[ERRO] Pasta não encontrada: {sim_results_dir}")
        return

    print("==================================================================")
    print(" INICIANDO FAXINA DE RESULTADOS (FITNESS < 0.75)")
    print("==================================================================\n")

    # O restante do script continua exatamente igual...
    json_files = glob.glob(os.path.join(sim_results_dir, "*.json"))
    timestamps_to_delete = set()

    print("[1/3] Analisando o fitness dentro dos arquivos JSON...")
    for file in json_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            fitness = None
            if "best_individual_so_far" in data and "fitness" in data["best_individual_so_far"]:
                fitness = data["best_individual_so_far"]["fitness"]
            elif "fitness_history" in data and data["fitness_history"]:
                fitness = max(data["fitness_history"])
            
            if fitness is not None and fitness < 0.75:
                match = re.search(r'(\d{8}_\d{6})', os.path.basename(file))
                if match:
                    timestamps_to_delete.add(match.group(1))
        except Exception:
            pass

    if not timestamps_to_delete:
        print("\nNenhum experimento com fitness abaixo de 0.75 foi encontrado. Sua pasta já está limpa!")
        print("==================================================================")
        return

    print(f"=> Encontrados {len(timestamps_to_delete)} experimentos falhos.")
    
    print("\n[2/3] Mapeando arquivos satélites associados a esses timestamps...")
    all_files = glob.glob(os.path.join(sim_results_dir, "*"))
    files_to_delete = []

    for file in all_files:
        filename = os.path.basename(file)
        if "sweep_summary" in filename:
            continue
            
        for ts in timestamps_to_delete:
            if ts in filename:
                files_to_delete.append(file)
                break 

    print(f"=> Um total de {len(files_to_delete)} arquivos lixo (JSONs, CSVs, PNGs, etc.) serão apagados.")

    print("\n[3/3] Execução de Limpeza")
    confirm = input("Tem certeza que deseja DELETAR PERMANENTEMENTE esses arquivos? (s/n): ")
    
    if confirm.lower().strip() == 's':
        deleted_count = 0
        for file in files_to_delete:
            try:
                os.remove(file)
                deleted_count += 1
            except Exception as e:
                print(f"  -> Erro ao deletar {os.path.basename(file)}: {e}")
        
        print("\n==================================================================")
        print(f" SUCESSO! Faxina concluída. {deleted_count} arquivos foram removidos.")
        print("==================================================================\n")
    else:
        print("\n==================================================================")
        print(" CANCELADO! Nenhuma alteração foi feita nos seus arquivos.")
        print("==================================================================\n")

if __name__ == "__main__":
    cleanup_failed_experiments()