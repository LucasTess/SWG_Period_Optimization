# experiment_end.py

import os
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np

def record_experiment_results(
    output_directory, 
    optimizer_instance, 
    experiment_start_time,

    Lambda_range, DC_range, w_range, height_range,

    generations_processed
):
    """
    Registra os resultados atuais do experimento em arquivos JSON e PNG.
    Os nomes dos arquivos são baseados no timestamp de início do experimento,
    permitindo que sejam sobrescritos durante a execução para salvar o progresso.
    """
    # --- Gera o nome do arquivo baseado no INÍCIO do experimento ---
    # Isso garante que o nome seja o mesmo durante toda a execução.
    timestamp_str = experiment_start_time.strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_directory, f"experiment_results_{timestamp_str}.json")
    plot_path = os.path.join(output_directory, f"fitness_history_{timestamp_str}.png")

    # --- Prepara os dados para o arquivo JSON ---
    current_time = datetime.datetime.now()
    duration = current_time - experiment_start_time

    results_data = {
        "experiment_start_time": experiment_start_time.isoformat(),
        "last_update": current_time.isoformat(), # Mostra quando foi a última atualização
        "current_duration": str(duration),
        "generations_processed": generations_processed,
        "population_size": optimizer_instance.population_size,
        "mutation_rate": optimizer_instance.mutation_rate,
        "max_generations_set": optimizer_instance.generations,
        "best_individual_so_far": optimizer_instance.best_individual,
        "best_fitness_so_far": optimizer_instance.best_fitness,
        "parameter_ranges": {
            "Lambda": Lambda_range,
            "DC": DC_range,
            "w": w_range,
            "height": height_range
        },
        "fitness_history": optimizer_instance.fitness_history
    }

    # --- Salva o arquivo JSON (sobrescrevendo o anterior) ---
    try:
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=4)
    except Exception as e:
        print(f"!!! Erro ao salvar/atualizar resultados do experimento: {e}")

    # --- Plota e salva o gráfico de fitness (sobrescrevendo o anterior) ---
    if optimizer_instance.fitness_history:
        plt.figure(figsize=(10, 6))
        generations = range(1, len(optimizer_instance.fitness_history) + 1)
        plt.plot(generations, optimizer_instance.fitness_history, marker='o', linestyle='-')
        plt.title(f'Histórico de Fitness (Atualizado em: {current_time.strftime("%H:%M:%S")})')
        plt.xlabel('Geração')
        plt.ylabel('Melhor Fitness')
        plt.grid(True)
        try:
            plt.savefig(plot_path)
        except Exception as e:
            print(f"!!! Erro ao salvar/atualizar gráfico de fitness: {e}")
        finally:
            plt.close() # Libera memória
    else:
        print("Nenhum histórico de fitness para plotar.")