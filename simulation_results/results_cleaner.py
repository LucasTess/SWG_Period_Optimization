import os
import glob
import json
import re
import shutil  # <-- Necessário para deletar diretórios inteiros

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

    # [MODIFICAÇÃO] Busca recursiva para encontrar os JSONs dentro das subpastas
    json_files = glob.glob(os.path.join(sim_results_dir, "**", "*.json"), recursive=True)
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
    
    print("\n[2/3] Mapeando arquivos satélites e pastas associadas a esses timestamps...")
    targets_to_delete = []

    # [MODIFICAÇÃO] Vai direto nos alvos específicos (A pasta do teste e o CSV bruto na raiz)
    for ts in timestamps_to_delete:
        # 1. Procura a subpasta
        folder_path = os.path.join(sim_results_dir, f"results_{ts}")
        if os.path.exists(folder_path):
            targets_to_delete.append(folder_path)
            
        # 2. Procura o CSV bruto na raiz
        csv_pattern = os.path.join(sim_results_dir, f"*{ts}_full_data.csv")
        csv_matches = glob.glob(csv_pattern)
        targets_to_delete.extend(csv_matches)

    print(f"=> Um total de {len(targets_to_delete)} itens (pastas e arquivos CSV) serão apagados.")

    print("\n[3/3] Execução de Limpeza")
    confirm = input("Tem certeza que deseja DELETAR PERMANENTEMENTE esses arquivos e pastas? (s/n): ")
    
    if confirm.lower().strip() == 's':
        deleted_count = 0
        for target in targets_to_delete:
            try:
                if os.path.isdir(target):
                    shutil.rmtree(target) # Apaga a pasta com tudo dentro
                else:
                    os.remove(target)     # Apaga o arquivo solto (CSV)
                deleted_count += 1
            except Exception as e:
                print(f"  -> Erro ao deletar {os.path.basename(target)}: {e}")
        
        print("\n==================================================================")
        print(f" SUCESSO! Faxina concluída. {deleted_count} itens foram removidos.")
        print("==================================================================\n")
    else:
        print("\n==================================================================")
        print(" CANCELADO! Nenhuma alteração foi feita nos seus arquivos.")
        print("==================================================================\n")

if __name__ == "__main__":
    cleanup_failed_experiments()