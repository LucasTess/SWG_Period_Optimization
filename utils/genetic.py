import random
import numpy as np

class GeneticOptimizer:
    def __init__(self, population_size, mutation_rate, generations,
                 Lambda_range, DC_range, w_range, w_c_range, N_range): # <--- MUDANÇA
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        
        # Estrutura de param_ranges atualizada para incluir tipo
        self.param_ranges = {
            'Lambda': {'range': Lambda_range, 'type': 'float'},
            'DC':     {'range': DC_range,     'type': 'float'},
            'w':      {'range': w_range,      'type': 'float'},
            'w_c':    {'range': w_c_range,    'type': 'float'}, # Novo
            'N':      {'range': N_range,      'type': 'int'}    # Novo
        }
        
        self.population = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.fitness_history = [] 

        # Parâmetros de referência atualizados
        self.reference_params = {
            'Lambda': 0.3e-6,
            'DC': 0.5,
            'w': 0.5e-6,
            'w_c': 0.25e-6, # w_c < w
            'N': 100
        }
        
        # Amplitudes de mutação inicial atualizadas
        self.initial_mutation_amplitude = {
            'Lambda': 0.5 * self.reference_params['Lambda'],
            'DC': 0.5 * self.reference_params['DC'],
            'w': 0.5 * self.reference_params['w'],
            'w_c': 0.5 * self.reference_params['w_c'],
            'N': 0.5 * self.reference_params['N']
        }
        
        # Steps de mutação local atualizados
        self.local_mutation_step = {}
        for param, info in self.param_ranges.items():
            param_range_width = info['range'][1] - info['range'][0]
            step = param_range_width * 0.05
            if info['type'] == 'int':
                # Garante que o step de mutação para inteiros seja pelo menos 1
                self.local_mutation_step[param] = max(1, int(round(step)))
            else:
                self.local_mutation_step[param] = step


    def _constrain_param(self, param_name, value):
        """Limita um valor com base nos ranges ESTÁTICOS e tipo em param_ranges."""
        param_info = self.param_ranges[param_name]
        min_val, max_val = param_info['range']
        
        constrained_val = max(min_val, min(max_val, value))
        
        if param_info['type'] == 'int':
            return int(round(constrained_val))
        return constrained_val

    def _enforce_dependent_constraints(self, chromosome):
        """
        Garante que as restrições DEPENDENTES (ex: w_c < w) sejam satisfeitas.
        Esta função é chamada APÓS a criação, crossover ou mutação.
        """
        # Restrição 1: w_c nunca pode ser maior que 0.8 * w
        w_c_max_allowed = chromosome['w'] * 0.8
        if chromosome['w_c'] > w_c_max_allowed:
            chromosome['w_c'] = w_c_max_allowed
            
        # Restrição 2: w_c nunca pode ser menor que o limite inferior definido (1e-6)
        # (Isso é necessário caso 0.8*w seja menor que o limite inferior)
        w_c_min_allowed = self.param_ranges['w_c']['range'][0]
        if chromosome['w_c'] < w_c_min_allowed:
            chromosome['w_c'] = w_c_min_allowed
            
        return chromosome


    def create_chromosome(self, reference_based=False):
        chromosome = {}
        if reference_based:
            for param in self.param_ranges.keys():
                ref_val = self.reference_params[param]
                variation_amplitude = self.initial_mutation_amplitude[param]
                val = random.uniform(ref_val - variation_amplitude, ref_val + variation_amplitude)
                chromosome[param] = self._constrain_param(param, val)
        else:
            # Versão corrigida: garante que mesmo aleatórios sejam limitados e do tipo certo
            for param, info in self.param_ranges.items():
                val = random.uniform(*info['range'])
                chromosome[param] = self._constrain_param(param, val)
        
        # Aplica as restrições dependentes ANTES de retornar
        chromosome = self._enforce_dependent_constraints(chromosome)
        return chromosome


    def initialize_population(self):
        self.population = []
        num_ref_based = self.population_size // 2
        num_random = self.population_size - num_ref_based

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
        # A seleção por torneio permanece a mesma
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
        
        # Garante que os filhos também obedeçam às restrições
        child1 = self._enforce_dependent_constraints(child1)
        child2 = self._enforce_dependent_constraints(child2)
        
        return child1, child2


    def mutate(self, chromosome, mutation_type='local'):
        if random.random() < self.mutation_rate:
            param_to_mutate = random.choice(list(self.param_ranges.keys()))
            param_info = self.param_ranges[param_to_mutate]
            current_val = chromosome[param_to_mutate]

            if mutation_type == 'local':
                mutation_step = self.local_mutation_step[param_to_mutate]
                new_val = current_val + random.uniform(-mutation_step, mutation_step)
            elif mutation_type == 'global':
                new_val = random.uniform(*param_info['range'])
            else: # Fallback
                mutation_step = self.local_mutation_step[param_to_mutate]
                new_val = current_val + random.uniform(-mutation_step, mutation_step)
            
            # Limita o valor mutado (e força o tipo, ex: int)
            chromosome[param_to_mutate] = self._constrain_param(param_to_mutate, new_val)
            
            # Garante que a mutação não violou restrições dependentes
            chromosome = self._enforce_dependent_constraints(chromosome)
        
        return chromosome



    def evolve(self, current_generation_fitness):
            if len(current_generation_fitness) != len(self.population):
                raise ValueError("O número de resultados de fitness não corresponde ao tamanho da população.")

            current_generation_best_individual = None
            current_generation_best_fitness = -float('inf')

            for i, individual in enumerate(self.population):
                individual_fitness = self.calculate_fitness(current_generation_fitness[i])
                individual['fitness'] = individual_fitness

                if individual_fitness > current_generation_best_fitness:
                    current_generation_best_fitness = individual_fitness
                    current_generation_best_individual = individual

                if individual_fitness > self.best_fitness:
                    self.best_fitness = individual_fitness
                    self.best_individual = {k: individual[k] for k in self.param_ranges.keys()}
                    self.best_individual['fitness'] = self.best_fitness

            self.fitness_history.append(self.best_fitness) 

            # Elitismo: o melhor indivíduo global sobrevive
            new_population = []
            if self.best_individual:
                # Copia o cromossomo de elite
                elite_chromosome = {k: self.best_individual[k] for k in self.param_ranges.keys()}
                new_population.append(elite_chromosome)

            # Preenche o resto da população
            num_to_generate = self.population_size - len(new_population)
            
            for _ in range(num_to_generate):
                parent1, parent2 = self.select_parents()
                child = random.choice(self.crossover(parent1, parent2))
                
                if random.random() < 0.5: 
                    child = self.mutate(child, mutation_type='local')
                else:
                    child = self.mutate(child, mutation_type='global')

                new_population.append(child)

            random.shuffle(new_population)
            self.population = new_population

            # Retorna a população limpa, pronta para a próxima simulação
            return [{k: chrom[k] for k in self.param_ranges.keys()} for chrom in self.population]