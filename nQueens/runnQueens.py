import mlrose
import numpy as np

from nQueens.nqueensUtil import provide_max_function


def run_ro_nqueens(state_size, initial_state):
    problem = mlrose.DiscreteOpt(length=state_size, fitness_fn=provide_max_function(),
                                 maximize=True, max_val=state_size)
    # Use randomized hill climbing for solution
    best_state, best_fitness = mlrose.random_hill_climb(problem=problem, init_state=initial_state, max_iters=1000,
                                                        random_state=50);
    print('The best state found is: ', best_state)
    print('The fitness at the best state is: ', best_fitness)
