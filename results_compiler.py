import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_best_fitness_from_json(results_directory):
    """
    Lê todos os arquivos .json em um diretório, extrai o valor de 'best_fitness',
    calcula estatísticas e gera um boxplot da distribuição desses valores.

    Args:
        results_directory (str): O caminho para a pasta que contém os arquivos .json.
    """
    
    fitness_values = []
    
    print(f"Analisando arquivos .json no diretório: '{results_directory}'\n")
    
    if not os.path.isdir(results_directory):
        print(f"Erro: O diretório '{results_directory}' não foi encontrado.")
        return

    for filename in os.listdir(results_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(results_directory, filename)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    if 'best_fitness_so_far' in data and data['best_fitness_so_far'] is not None:
                        fitness = data['best_fitness_so_far']
                        fitness_values.append(fitness)
                        print(f"  - Arquivo '{filename}': best_fitness = {fitness:.4e}")
                    else:
                        print(f"  - AVISO: A chave 'best_fitness' não foi encontrada ou é nula no arquivo '{filename}'.")
                        
            except json.JSONDecodeError:
                print(f"  - ERRO: O arquivo '{filename}' não é um JSON válido.")
            except Exception as e:
                print(f"  - ERRO: Ocorreu um erro ao processar o arquivo '{filename}': {e}")

    # --- CÁLCULO E RESULTADOS ESTATÍSTICOS ---
    
    if not fitness_values:
        print("\nNenhum valor de 'best_fitness' foi encontrado para análise.")
        return

    fitness_array = np.array(fitness_values)
    
    mean_fitness = np.mean(fitness_array)
    std_dev_fitness = np.std(fitness_array)
    median_fitness = np.median(fitness_array) # Mediana é o centro do boxplot
    num_files = len(fitness_array)
    
    print("\n--- Resultados da Análise ---")
    print(f"Número de experimentos analisados: {num_files}")
    print(f"Média do best_fitness:           {mean_fitness:.4e}")
    print(f"Mediana do best_fitness:          {median_fitness:.4e}")
    print(f"Desvio Padrão do best_fitness:    {std_dev_fitness:.4e}")
    print("-----------------------------\n")

    # --- GERAÇÃO E SALVAMENTO DO BOXPLOT ---
    
    print("Gerando boxplot da distribuição de fitness...")
    
    # Define o caminho de saída para a imagem no mesmo diretório
    boxplot_output_path = os.path.join(results_directory, "fitness_distribution_boxplot.png")
    
    # Cria a figura do gráfico com um bom tamanho
    plt.figure(figsize=(8, 10))
    
    # Usa a biblioteca Seaborn para criar um boxplot visualmente agradável
    sns.boxplot(y=fitness_array, palette="viridis", width=0.4)
    sns.swarmplot(y=fitness_array, color=".25") # Adiciona os pontos individuais sobre o gráfico

    # Adiciona títulos e labels para clareza
    plt.title("Distribuição dos Melhores Fitness Entre os Experimentos", fontsize=16, pad=20)
    plt.ylabel("Valor do Best Fitness (delta_amp)", fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7) # Adiciona uma grade horizontal
    
    # Salva a figura no arquivo e depois fecha para liberar memória
    try:
        plt.savefig(boxplot_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"-> Boxplot salvo com sucesso em: {boxplot_output_path}")
    except Exception as e:
        print(f"!!! Erro ao salvar o boxplot: {e}")
    # --- FIM DO BLOCO DO BOXPLOT ---


if __name__ == '__main__':
    # --- CONFIGURE O CAMINHO AQUI ---
    # Altere este caminho para a sua pasta 'simulation_results'
    directory_to_analyze = "C:\\Users\\User04\\Documents\\metamaterial_guide_otimization\\simulation_results"
    
    analyze_best_fitness_from_json(directory_to_analyze)