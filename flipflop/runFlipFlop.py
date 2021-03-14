from datetime import datetime

import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose

def run_flip_flop(initial_state, vector_dimension):
    fitness = mlrose.FlipFlop()
    problem = mlrose.DiscreteOpt(length=vector_dimension, fitness_fn=fitness, maximize=True)

    results = {}

    print("Running randomized hill climbing")

    start = datetime.now()
    rhc_results, rhc_fitness = mlrose.random_hill_climb(problem=problem, init_state=initial_state, max_attempts=100, max_iters=1000);
    end = datetime.now()
    results['RHC'] = [rhc_fitness, (end-start).total_seconds()]

    print("Running simulated annealing")
    start = datetime.now()
    sa_results, sa_fitness =  mlrose.simulated_annealing(problem, schedule=mlrose.ExpDecay(), init_state=initial_state, max_attempts=100, max_iters=1000);
    end = datetime.now()
    results['SA'] = [sa_fitness, (end-start).total_seconds()]

    print("Running genetic algorithm")
    start = datetime.now()
    ga_results, ga_fitness =  mlrose.genetic_alg(problem, max_iters=200, mutation_prob=0.5, max_attempts=200)

    end = datetime.now()
    results['GA'] = [ga_fitness, (end-start).total_seconds()]

    print("Running MIMIC")
    start = datetime.now()
    mimic_results, mimic_fitness = mlrose.mimic(problem, max_iters=200,keep_pct=.5, max_attempts=200)
    end = datetime.now()
    results['MIMIC'] = [mimic_fitness, (end-start).total_seconds()]

    return results