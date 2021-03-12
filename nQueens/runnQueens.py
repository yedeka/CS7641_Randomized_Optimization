import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose

from nQueens.nqueensUtil import provide_max_function


def run_ro_nqueens(state_size, initial_state, max_iters):
    problem = mlrose.DiscreteOpt(length=state_size, fitness_fn=provide_max_function(),
                                 maximize=True, max_val=state_size)
    # Use randomized hill climbing for solution
    hcbest_state, hcbest_fitness = mlrose.random_hill_climb(problem=problem, init_state=initial_state, max_iters=max_iters);
    print('results of Random Hill Climbing')
    print('-------------------------------')
    print('-------------------------------')
    print('The best state found is: ', hcbest_state)
    print('The fitness at the best state is: ', hcbest_fitness)
    print('--------------------------------------------------')
    print('results of Simulated Annealing')
    print('-------------------------------')
    print('-------------------------------')
    # Use simulated annealing for solution
    sabest_state, sabest_fitness = mlrose.simulated_annealing(problem, schedule=mlrose.ExpDecay(), init_state=initial_state, max_iters=max_iters);
    print('The best state found is: ', sabest_state)
    print('The fitness at the best state is: ', sabest_fitness)
    print('--------------------------------------------------')
    print('results of Genetic Algorithm')
    print('-------------------------------')
    print('-------------------------------')
    # Use genetic algorithm for solution
    gabest_state, gabest_fitness = mlrose.genetic_alg(problem, max_iters=max_iters);
    print('The best state found is: ', gabest_state)
    print('The fitness at the best state is: ', gabest_fitness)
    print('--------------------------------------------------')
    # Use mimc for solution
    print('results of mimic')
    print('-------------------------------')
    print('-------------------------------')
    mbest_state, mbest_fitness = mlrose.mimic(problem, max_iters=max_iters);
    print('The best state found is: ', mbest_state)
    print('The fitness at the best state is: ', mbest_fitness)
    print('--------------------------------------------------')
    return {
        'hill_climbing_fitness': hcbest_fitness,
        'simulated_annealing_fitness': sabest_fitness,
        'genetic_algorithm_fitness': gabest_fitness,
        'mimic_fitness': mbest_fitness
    }

