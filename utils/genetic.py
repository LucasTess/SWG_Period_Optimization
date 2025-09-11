import random
import numpy as np

class GeneticOptimizer:
    def __init__(self, population_size, mutation_rate, generations, # 'generations' já está ok
                 Lambda_range, DC_range, w_range, height_range):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations # Nome da variável para o número máximo de gerações
        self.param_ranges = {
            'Lambda': Lambda_range,
            'DC': DC_range,
            'w': w_range,
            'height': height_range
        }
        self.population = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.fitness_history = [] # <--- NOVO: Inicializa o histórico de fitness

        self.reference_params = {
            'Lambda': 0.15e-6,
            'DC': 0.5e-6,
            'w': 0.15e-6,
            'height': 0.22e-6
        }
        self.initial_mutation_amplitude = {
            'Lambda': 0.5 * self.reference_params['Lambda'],
            'DC': 0.5 * self.reference_params['DC'],
            'w': 0.5 * self.reference_params['w'],
            'height': 0.5 * self.reference_params['height']
        }
        # Define a amplitude da mutação "local" como uma porcentagem do range total
        self.local_mutation_step = {
            param: (self.param_ranges[param][1] - self.param_ranges[param][0]) * 0.05
            for param in self.param_ranges
        }


    def _constrain_param(self, param_name, value):
        min_val, max_val = self.param_ranges[param_name]
        return max(min_val, min(max_val, value))


    def create_chromosome(self, reference_based=False):
        chromosome = {}
        if reference_based:
            for param, ref_val in self.reference_params.items():
                variation_amplitude = self.initial_mutation_amplitude[param]
                val = random.uniform(ref_val - variation_amplitude, ref_val + variation_amplitude)
                chromosome[param] = self._constrain_param(param, val)
        else:
            for param in self.param_ranges:
                chromosome[param] = random.uniform(*self.param_ranges[param])
        return chromosome


    def initialize_population(self):
        self.population = []
        num_ref_based = self.population_size // 2  # Metade da população baseada em referência
        num_random = self.population_size - num_ref_based # A outra metade aleatória

        for _ in range(num_ref_based):
            self.population.append(self.create_chromosome(reference_based=True))

        for _ in range(num_random):
            self.population.append(self.create_chromosome(reference_based=False))

        random.shuffle(self.population)


    def calculate_fitness(self, delta_amp):
        if np.isinf(delta_amp) or np.isnan(delta_amp):
            return -float('inf')
        return delta_amp


    def select_parents(self):
        pool = random.sample(self.population, min(5, len(self.population)))
        parent1 = max(pool, key=lambda x: x.get('fitness', -float('inf')))

        pool = random.sample(self.population, min(5, len(self.population)))
        parent2 = max(pool, key=lambda x: x.get('fitness', -float('inf')))

        return parent1, parent2


    def crossover(self, parent1, parent2):
        child1 = {}
        child2 = {}
        keys = list(self.param_ranges.keys())
        crossover_point = random.randint(1, len(keys) - 1)

        for i, key in enumerate(keys):
            if i < crossover_point:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        
        return child1, child2


    def mutate(self, chromosome, mutation_type='local'):
        if random.random() < self.mutation_rate:
            param_to_mutate = random.choice(list(self.param_ranges.keys()))
            param_range = self.param_ranges[param_to_mutate]
            current_val = chromosome[param_to_mutate]

            if mutation_type == 'local':
                mutation_step = self.local_mutation_step[param_to_mutate]
                new_val = current_val + random.uniform(-mutation_step, mutation_step)
            elif mutation_type == 'global':
                new_val = random.uniform(*param_range)
            else: # Fallback to local if unknown type
                mutation_step = self.local_mutation_step[param_to_mutate]
                new_val = current_val + random.uniform(-mutation_step, mutation_step)
            
            chromosome[param_to_mutate] = self._constrain_param(param_to_mutate, new_val)
        return chromosome



    def evolve(self, current_generation_fitness):
            if len(current_generation_fitness) != len(self.population):

                raise ValueError("O número de resultados de delta_amp não corresponde ao tamanho da população.")

            current_generation_best_individual = None
            current_generation_best_fitness = -float('inf')

            for i, individual in enumerate(self.population):
                individual_fitness = self.calculate_fitness(current_generation_fitness[i])
                individual['fitness'] = individual_fitness

                # 1. Encontra o melhor indivíduo da GERAÇÃO ATUAL
                if individual_fitness > current_generation_best_fitness:
                    current_generation_best_fitness = individual_fitness
                    current_generation_best_individual = individual

                # Atualiza o melhor indivíduo GLOBAL, se o atual for melhor
                if individual_fitness > self.best_fitness:
                    self.best_fitness = individual_fitness
                    self.best_individual = {k: individual[k] for k in self.param_ranges.keys()}
                    self.best_individual['fitness'] = self.best_fitness

            self.fitness_history.append(self.best_fitness) # Usa o melhor fitness GLOBAL

            # 2. IMPLEMENTAÇÃO DO ELITISMO - Garante que o melhor indivíduo global sempre sobreviva.
            new_population = []
            if self.best_individual:
                elite_chromosome = {k: self.best_individual[k] for k in self.param_ranges.keys()}
                new_population.append(elite_chromosome)

            # Preenche o restante da nova população
            num_to_generate = self.population_size - len(new_population)
            
            # A lógica de geração de filhos e mutação (local/global) permanece a mesma
            for _ in range(num_to_generate):
                parent1, parent2 = self.select_parents()
                child = random.choice(self.crossover(parent1, parent2))
                
                # Decide se a mutação será local ou global
                if random.random() < 0.5: # 50% de chance para cada tipo de mutação
                    child = self.mutate(child, mutation_type='local')
                else:
                    child = self.mutate(child, mutation_type='global')

                new_population.append(child)

            random.shuffle(new_population)
            self.population = new_population

            return [{k: chrom[k] for k in self.param_ranges.keys()} for chrom in self.population]