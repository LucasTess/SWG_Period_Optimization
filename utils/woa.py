# utils/woa.py
import random
import math
import numpy as np

class WhaleOptimizer:
    def __init__(self, population_size, mutation_rate, generations,
                 Lambda_range, DC_range, w_range, w_c_range, N_range):
        self.population_size = population_size
        self.generations = generations
        # Mantido por compatibilidade com a assinatura do supervisor/GA
        self.mutation_rate = mutation_rate 
        
        self.current_generation = 0 

        self.param_ranges = {
            'Lambda': {'range': Lambda_range, 'type': 'float'},
            'DC':     {'range': DC_range,     'type': 'float'},
            'w':      {'range': w_range,      'type': 'float'},
            'w_c':    {'range': w_c_range,    'type': 'float'},
            'N':      {'range': N_range,      'type': 'int'}   
        }
        
        self.population = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.fitness_history = [] 

    def _constrain_param(self, param_name, value):
        """Limita um valor com base nos ranges ESTÁTICOS e tipo em param_ranges."""
        param_info = self.param_ranges[param_name]
        min_val, max_val = param_info['range']
        
        constrained_val = max(min_val, min(max_val, value))
        
        if param_info['type'] == 'int':
            return int(round(constrained_val))
        return constrained_val

    def _enforce_dependent_constraints(self, chromosome):
        """Garante que as restrições DEPENDENTES sejam satisfeitas."""
        w_c_max_allowed = chromosome['w'] * 0.8
        if chromosome['w_c'] > w_c_max_allowed:
            chromosome['w_c'] = w_c_max_allowed
            
        w_c_min_allowed = self.param_ranges['w_c']['range'][0]
        if chromosome['w_c'] < w_c_min_allowed:
            chromosome['w_c'] = w_c_min_allowed
            
        return chromosome

    def create_whale(self):
        """Cria uma baleia (solução) aleatória respeitando os limites."""
        chromosome = {}
        for param, info in self.param_ranges.items():
            val = random.uniform(*info['range'])
            chromosome[param] = self._constrain_param(param, val)
        
        chromosome = self._enforce_dependent_constraints(chromosome)
        return chromosome

    def initialize_population(self):
        """No WOA, inicializamos as baleias de forma puramente aleatória."""
        self.population = [self.create_whale() for _ in range(self.population_size)]

    def calculate_fitness(self, score):
        if np.isinf(score) or np.isnan(score):
            return -float('inf')
        return score

    def evolve(self, current_generation_fitness):
        """Atualiza a posição do cardume de baleias com Elitismo Estrito."""
        if len(current_generation_fitness) != len(self.population):
            raise ValueError("O número de resultados não corresponde à população.")

        # 1. Avalia o cardume e encontra a melhor baleia global
        for i, individual in enumerate(self.population):
            individual_fitness = self.calculate_fitness(current_generation_fitness[i])
            individual['fitness'] = individual_fitness

            if individual_fitness > self.best_fitness:
                self.best_fitness = individual_fitness
                self.best_individual = {k: individual[k] for k in self.param_ranges.keys()}
                self.best_individual['fitness'] = self.best_fitness

        self.fitness_history.append(self.best_fitness) 

        # 2. Dinâmica do WOA: O parâmetro 'a' decresce linearmente de 2 a 0
        a = 2.0 - (self.current_generation * (2.0 / self.generations))

        new_population = []

        # --- CORREÇÃO: ELITISMO ESTRITO ---
        # Garante que a geometria perfeita seja repassada intacta para a próxima geração
        if self.best_individual:
            elite_whale = {k: self.best_individual[k] for k in self.param_ranges.keys()}
            new_population.append(elite_whale)

        # 3. Atualiza a posição do resto do cardume
        for i in range(self.population_size):
            # Se já preenchemos a população (devido à baleia de elite), paramos o loop
            if len(new_population) >= self.population_size:
                break
                
            current_whale = self.population[i]
            new_whale = {}
            
            p = random.random()
            r1 = random.random()
            r2 = random.random()
            
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            
            b = 1.0 # Constante que define a forma da espiral
            l = random.uniform(-1.0, 1.0)

            # Equações de Movimento do WOA
            if p < 0.5:
                if abs(A) < 1.0:
                    # Mecanismo de cerco à presa (Explotação)
                    for param in self.param_ranges.keys():
                        D = abs(C * self.best_individual[param] - current_whale[param])
                        new_whale[param] = self.best_individual[param] - A * D
                else:
                    # Busca global por uma nova presa (Exploração)
                    random_whale = random.choice(self.population)
                    for param in self.param_ranges.keys():
                        D = abs(C * random_whale[param] - current_whale[param])
                        new_whale[param] = random_whale[param] - A * D
            else:
                # Atualização em espiral ao redor da melhor presa (Explotação forte)
                for param in self.param_ranges.keys():
                    D_prime = abs(self.best_individual[param] - current_whale[param])
                    new_whale[param] = D_prime * math.exp(b * l) * math.cos(2.0 * math.pi * l) + self.best_individual[param]

            # 4. Limita a nova posição aos limites físicos de fabricação
            for param in self.param_ranges.keys():
                new_whale[param] = self._constrain_param(param, new_whale[param])
                
            new_whale = self._enforce_dependent_constraints(new_whale)
            new_population.append(new_whale)

        self.population = new_population
        self.current_generation += 1

        return [{k: chrom[k] for k in self.param_ranges.keys()} for chrom in self.population]