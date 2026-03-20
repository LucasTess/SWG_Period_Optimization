import random
import numpy as np

class GeneticOptimizer:
    def __init__(self, population_size, mutation_rate, generations, ranges_dict):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        
        # Identifica o modo (uniform ou apodized)
        self.mode = ranges_dict.get('mode', 'uniform')
        self.ranges_raw = ranges_dict

        # Define dinamicamente quais genes serão otimizados
        self.param_ranges = {
            'Lambda': {'range': ranges_dict['Lambda_range'], 'type': 'float'},
            'DC':     {'range': ranges_dict['DC_range'],     'type': 'float'},
            'w':      {'range': ranges_dict['w_range'],      'type': 'float'},
            'N':      {'range': ranges_dict['N_range'],      'type': 'int'}
        }
        
        # w_c é tratado com um range dinâmico baseado no w_c_range_max_ratio
        # mas definimos uma entrada aqui para facilitar a lógica de mutação/limitação
        max_w = ranges_dict['w_range'][1]
        self.param_ranges['w_c'] = {'range': (1e-7, max_w), 'type': 'float'}

        # Adiciona os genes de apodização se estiver no modo correto
        if self.mode == "apodized":
            self.param_ranges['H'] = {'range': ranges_dict['H_range'], 'type': 'int'}
            self.param_ranges['alpha_param'] = {'range': ranges_dict['alpha_range'], 'type': 'float'}
            self.param_ranges['beta'] = {'range': ranges_dict['beta_range'], 'type': 'float'}
            self.param_ranges['S'] = {'range': ranges_dict['S_range'], 'type': 'int'}
        self.population = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.fitness_history = [] 

        # Parâmetros de referência 
        self.reference_params = {
            'Lambda': 0.3e-6, 'DC': 0.5, 'w': 0.5e-6, 'w_c': 0.25e-6, 'N': 100,
            'H': 1, 'alpha_param': 1.0, 'beta': 1.57, 'S': 0
        }
        
        # Steps de mutação local (5% do range total)
        self.local_mutation_step = {}
        for param, info in self.param_ranges.items():
            param_range_width = info['range'][1] - info['range'][0]
            step = param_range_width * 0.05
            if info['type'] == 'int':
                self.local_mutation_step[param] = max(1, int(round(step)))
            else:
                self.local_mutation_step[param] = step

    def _constrain_param(self, param_name, value):
        """Limita um valor com base nos ranges e tipos."""
        param_info = self.param_ranges[param_name]
        min_val, max_val = param_info['range']
        constrained_val = max(min_val, min(max_val, value))
        
        if param_info['type'] == 'int':
            return int(round(constrained_val))
        return constrained_val

    def _enforce_dependent_constraints(self, chromosome):
        """Garante restrições físicas entre variáveis."""
        # 1. w_c < ratio * w
        w_c_max = chromosome['w'] * self.ranges_raw.get('w_c_range_max_ratio', 0.8)
        if chromosome['w_c'] > w_c_max:
            chromosome['w_c'] = w_c_max

        # O delta_s_max foi removido daqui pois a restrição física (Lambda * DC) 
        # agora é calculada nativamente dentro do script LSF do Lumerical.
            
        return chromosome

    def create_chromosome(self, reference_based=False):
        chromosome = {}
        if reference_based:
            for param in self.param_ranges.keys():
                ref_val = self.reference_params.get(param, 0)
                # Variação de 20% em torno da referência para diversidade controlada
                val = random.uniform(ref_val * 0.8, ref_val * 1.2)
                chromosome[param] = self._constrain_param(param, val)
        else:
            for param, info in self.param_ranges.items():
                val = random.uniform(*info['range'])
                chromosome[param] = self._constrain_param(param, val)
        
        return self._enforce_dependent_constraints(chromosome)

    def initialize_population(self):
        self.population = []
        num_ref_based = self.population_size // 4 # Reduzi para 25% para dar mais espaço à exploração
        for _ in range(num_ref_based):
            self.population.append(self.create_chromosome(reference_based=True))
        for _ in range(self.population_size - num_ref_based):
            self.population.append(self.create_chromosome(reference_based=False))
        random.shuffle(self.population)

    def calculate_fitness(self, score):
        if np.isinf(score) or np.isnan(score):
            return -1e10 # Penalidade pesada mas finita
        return score

    def select_parents(self):
        # Torneio
        pool1 = random.sample(self.population, min(4, len(self.population)))
        parent1 = max(pool1, key=lambda x: x.get('fitness', -1e10))
        pool2 = random.sample(self.population, min(4, len(self.population)))
        parent2 = max(pool2, key=lambda x: x.get('fitness', -1e10))
        return parent1, parent2

    def crossover(self, parent1, parent2):
        child1, child2 = {}, {}
        keys = list(self.param_ranges.keys())
        cp = random.randint(1, len(keys) - 1)
        for i, key in enumerate(keys):
            if i < cp:
                child1[key], child2[key] = parent1[key], parent2[key]
            else:
                child1[key], child2[key] = parent2[key], parent1[key]
        return self._enforce_dependent_constraints(child1), self._enforce_dependent_constraints(child2)

    def mutate(self, chromosome, mutation_type='local'):
        if random.random() < self.mutation_rate:
            param = random.choice(list(self.param_ranges.keys()))
            if mutation_type == 'local':
                step = self.local_mutation_step[param]
                new_val = chromosome[param] + random.uniform(-step, step)
            else:
                new_val = random.uniform(*self.param_ranges[param]['range'])
            
            chromosome[param] = self._constrain_param(param, new_val)
            chromosome = self._enforce_dependent_constraints(chromosome)
        return chromosome

    def evolve(self, current_generation_fitness):
        if len(current_generation_fitness) != len(self.population):
            raise ValueError("Fitness array size mismatch.")

        # Atribui fitness e atualiza melhor global
        for i, individual in enumerate(self.population):
            fit = self.calculate_fitness(current_generation_fitness[i])
            individual['fitness'] = fit
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_individual = {k: v for k, v in individual.items()}

        self.fitness_history.append(self.best_fitness) 

        # Elitismo e nova geração
        new_population = []
        if self.best_individual:
            new_population.append({k: v for k, v in self.best_individual.items() if k != 'fitness'})

        while len(new_population) < self.population_size:
            p1, p2 = self.select_parents()
            c1, c2 = self.crossover(p1, p2)
            m_type = 'global' if random.random() < 0.2 else 'local'
            new_population.append(self.mutate(c1, m_type))
            if len(new_population) < self.population_size:
                new_population.append(self.mutate(c2, m_type))

        self.population = new_population
        # Retorna apenas os dicionários de genes, sem o campo 'fitness'
        return [{k: v for k, v in chrom.items() if k in self.param_ranges} for chrom in self.population]