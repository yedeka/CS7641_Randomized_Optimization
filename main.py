import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nQueens.runnQueens import run_ro_nqueens
from fourPeaks.runFourPeaks import run_four_peaks

def perform_plot_fourPeaks():
    # Perform four peaks experiment
    problem_size = 20
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_20 = run_four_peaks(init_state(), problem_size)
    print(results_20)
    problem_size = 40
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_40 = run_four_peaks(init_state(), problem_size)
    print(results_40)
    problem_size = 60
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_60 = run_four_peaks(init_state(), problem_size)
    print(results_60)
    problem_size = 80
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_80 = run_four_peaks(init_state(), problem_size)
    print(print(results_80))
    problem_size = 100
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_100 = run_four_peaks(init_state(), problem_size)
    print(results_100)
    data = {'Problem_Size': [20,40,60,80,100],
            'RHC': [results_20['RHC'][0], results_40['RHC'][0], results_60['RHC'][0], results_80['RHC'][0], results_100['RHC'][0]],
            'SA' : [results_20['SA'][0], results_40['SA'][0], results_60['SA'][0], results_80['SA'][0], results_100['SA'][0]],
            'GA' : [results_20['GA'][0], results_40['GA'][0], results_60['GA'][0], results_80['GA'][0], results_100['GA'][0]],
            'MIMIC' : [results_20['MIMIC'][0], results_40['MIMIC'][0], results_60['MIMIC'][0], results_80['MIMIC'][0], results_100['MIMIC'][0]],
            }

    df = pd.DataFrame(data, columns=['Problem_Size', 'RHC', 'SA','GA','MIMIC'])
    plt.plot(x =df['Problem_Size'], y=df['RHC'], label='Randomized Hill Climbing')
    #plt.plot(x=data['Problem_Size'], y=data['SA'], label='Simulated Annealing', marker='o', color='blue')
    #plt.plot(x=data['Problem_Size'], y=data['GA'], label='Genetic Algorithm', marker='o', color='yellow')
    #plt.plot(x=data['Problem_Size'], y=data['MIMIC'], label='MIMIC', marker='o', color='green')
    plt.xlabel('Problem Size')
    # Set the y axis label of the current axis.
    plt.ylabel('Fitness Values')
    # Set a title of the current axes.
    plt.title('Randomized Optimization for 4 peaks problem')
    plt.grid(True)
    plt.savefig('4 Peaks Performance.png')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define initial state
    '''init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    first_iters_fitness = run_ro_nqueens(8, init_state,1000)
    second_iters_fitness = run_ro_nqueens(8, init_state, 2000)
    third_iters_fitness = run_ro_nqueens(8, init_state, 3000)
    fourth_iters_fitness = run_ro_nqueens(8, init_state, 4000)
    fifth_iters_fitness = run_ro_nqueens(8, init_state, 5000)'''
    perform_plot_fourPeaks()
