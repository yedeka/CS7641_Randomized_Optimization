import numpy as np
from nQueens.runnQueens import run_ro_nqueens

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define initial state
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    first_iters_fitness = run_ro_nqueens(8, init_state,1000)
    second_iters_fitness = run_ro_nqueens(8, init_state, 2000)
    third_iters_fitness = run_ro_nqueens(8, init_state, 3000)
    fourth_iters_fitness = run_ro_nqueens(8, init_state, 4000)
    fifth_iters_fitness = run_ro_nqueens(8, init_state, 5000)
