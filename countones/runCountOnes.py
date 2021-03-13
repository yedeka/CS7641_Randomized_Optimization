from datetime import datetime

import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose

def run_one_max(initial_state, vector_dimension):
    fitness = mlrose.OneMax()
    problem = mlrose.DiscreteOpt(length=vector_dimension, fitness_fn=fitness, maximize=True)

    results = {}

    print("Running randomized hill climbing")

    start = datetime.now()
    rhc_results, rhc_fitness = mlrose.random_hill_climb(problem=problem, init_state=initial_state, max_attempts=100, max_iters=1000);
    end = datetime.now()
    results['RHC'] = [rhc_fitness, end-start]

    print("Running simulated annealing")
    start = datetime.now()
    sa_results, sa_fitness =  mlrose.simulated_annealing(problem, schedule=mlrose.ExpDecay(), init_state=initial_state, max_attempts=100, max_iters=1000);
    end = datetime.now()
    results['SA'] = [sa_fitness, end-start]

    print("Running genetic algorithm")
    start = datetime.now()
    ga_results, ga_fitness =  mlrose.genetic_alg(problem, max_attempts=100, max_iters=1000);
    end = datetime.now()
    results['GA'] = [ga_fitness, end-start]

    print("Running MIMIC")
    start = datetime.now()
    mimic_results, mimic_fitness = mlrose.mimic(problem, max_attempts=100, max_iters=1000);
    end = datetime.now()
    results['MIMIC'] = [mimic_fitness, end - start]

    return results