import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fourPeaks.runFourPeaks import run_four_peaks
from flipflop.runFlipFlop import run_flip_flop
from countones.runCountOnes import run_one_max

def perform_one_max():
    print("Running One Max experiment")
    # Perform flip flop experiment
    problem_size = 20
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_20 = run_one_max(init_state(), problem_size)
    print("One Max n =20")
    print(results_20)
    problem_size = 40
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_40 = run_one_max(init_state(), problem_size)
    print("One Max n =40")
    print(results_40)
    problem_size = 60
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_60 = run_one_max(init_state(), problem_size)
    print("One Max n =60")
    print(results_60)
    problem_size = 80
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_80 = run_one_max(init_state(), problem_size)
    print("One Max n =80")
    print(results_80)
    problem_size = 100
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_100 = run_one_max(init_state(), problem_size)
    print("One Max n =100")
    print(results_100)
    data = {'Problem_Size': [20, 40, 60, 80, 100],
            'RHC': [results_20['RHC'][0], results_40['RHC'][0], results_60['RHC'][0], results_80['RHC'][0],
                    results_100['RHC'][0]],
            'SA': [results_20['SA'][0], results_40['SA'][0], results_60['SA'][0], results_80['SA'][0],
                   results_100['SA'][0]],
            'GA': [results_20['GA'][0], results_40['GA'][0], results_60['GA'][0], results_80['GA'][0],
                   results_100['GA'][0]],
            'MIMIC': [results_20['MIMIC'][0], results_40['MIMIC'][0], results_60['MIMIC'][0], results_80['MIMIC'][0],
                      results_100['MIMIC'][0]],
            }
    df = pd.DataFrame(data, columns=['Problem_Size', 'RHC', 'SA', 'GA', 'MIMIC'])

    print("One Max results")
    print(df)

    ax = plt.gca()
    df.plot(kind='line', x='Problem_Size', y='RHC', marker='o', ax=ax)
    df.plot(kind='line', x='Problem_Size', y='SA', marker='o', ax=ax)
    df.plot(kind='line', x='Problem_Size', y='GA', marker='o', ax=ax)
    df.plot(kind='line', x='Problem_Size', y='MIMIC', marker='o', ax=ax)

    plt.xlabel('Problem Size')
    plt.ylabel('Fitness Values')
    plt.title('Randomized Optimization for one max problem')
    plt.grid(True)
    plt.savefig('One Max Performance.png')

def perform_flip_flop():
    print("Running flip flop experiment")
    # Perform flip flop experiment
    problem_size = 20
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_20 = run_flip_flop(init_state(), problem_size)
    print("Flip flop n =20")
    print(results_20)
    problem_size = 40
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_40 = run_flip_flop(init_state(), problem_size)
    print("Flip flop n =40")
    print(results_40)
    problem_size = 60
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_60 = run_flip_flop(init_state(), problem_size)
    print("Flip flop n =60")
    print(results_60)
    problem_size = 80
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_80 = run_flip_flop(init_state(), problem_size)
    print("Flip flop n =80")
    print(results_80)
    problem_size = 100
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_100 = run_flip_flop(init_state(), problem_size)
    print("Flip flop n =100")
    print(results_100)
    problem_size = 120
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_120 = run_flip_flop(init_state(), problem_size)
    print("Flip flop n =120")
    print(results_120)
    problem_size = 150
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_150 = run_flip_flop(init_state(), problem_size)
    print("Flip flop n =120")
    print(results_150)
    problem_size = 200
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_200 = run_flip_flop(init_state(), problem_size)
    print("Flip flop n =200")
    print(results_200)

    data = {'Problem_Size': [20, 40, 60, 80, 100, 120, 150, 200],
            'RHC': [results_20['RHC'][0], results_40['RHC'][0], results_60['RHC'][0], results_80['RHC'][0],
                    results_100['RHC'][0], results_120['RHC'][0], results_150['RHC'][0], results_200['RHC'][0]],
            'SA': [results_20['SA'][0], results_40['SA'][0], results_60['SA'][0], results_80['SA'][0],
                   results_100['SA'][0], results_120['RHC'][0], results_150['RHC'][0], results_200['RHC'][0]],
            'GA': [results_20['GA'][0], results_40['GA'][0], results_60['GA'][0], results_80['GA'][0],
                   results_100['GA'][0], results_120['RHC'][0], results_150['RHC'][0], results_200['RHC'][0]],
            'MIMIC': [results_20['MIMIC'][0], results_40['MIMIC'][0], results_60['MIMIC'][0], results_80['MIMIC'][0],
                      results_100['MIMIC'][0], results_120['RHC'][0], results_150['RHC'][0], results_200['RHC'][0]],
            }
    df = pd.DataFrame(data, columns=['Problem_Size', 'RHC', 'SA', 'GA', 'MIMIC'])

    print("Flip Flop results")
    print(df)

    ax = plt.gca()
    df.plot(kind='line', x='Problem_Size', y='RHC', marker='o', ax=ax)
    df.plot(kind='line', x='Problem_Size', y='SA', marker='o', ax=ax)
    df.plot(kind='line', x='Problem_Size', y='GA', marker='o', ax=ax)
    df.plot(kind='line', x='Problem_Size', y='MIMIC', marker='o', ax=ax)

    plt.xlabel('Problem Size')
    plt.ylabel('Fitness Values')
    plt.title('Randomized Optimization for flip flop problem')
    plt.grid(True)

    plt.savefig('Flip flop Performance.png')

def perform_plot_fourPeaks():
    print("Running 4 peaks experiment")
    # Perform four peaks experiment
    problem_size = 20
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_20 = run_four_peaks(init_state(), problem_size)
    print("4 Peaks n =20")
    print(results_20)
    problem_size = 40
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_40 = run_four_peaks(init_state(), problem_size)
    print("4 Peaks n =40")
    print(results_40)
    problem_size = 60
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_60 = run_four_peaks(init_state(), problem_size)
    print("4 Peaks n =60")
    print(results_60)
    problem_size = 80
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_80 = run_four_peaks(init_state(), problem_size)
    print("4 Peaks n =80")
    print(results_80)
    problem_size = 100
    init_state = lambda: np.random.randint(2, size=problem_size)
    results_100 = run_four_peaks(init_state(), problem_size)
    print("4 Peaks n =100")
    print(results_100)
    data = {'Problem_Size': [20,40,60,80,100],
            'RHC': [results_20['RHC'][0], results_40['RHC'][0], results_60['RHC'][0], results_80['RHC'][0], results_100['RHC'][0]],
            'SA' : [results_20['SA'][0], results_40['SA'][0], results_60['SA'][0], results_80['SA'][0], results_100['SA'][0]],
            'GA' : [results_20['GA'][0], results_40['GA'][0], results_60['GA'][0], results_80['GA'][0], results_100['GA'][0]],
            'MIMIC' : [results_20['MIMIC'][0], results_40['MIMIC'][0], results_60['MIMIC'][0], results_80['MIMIC'][0], results_100['MIMIC'][0]],
            }
    df = pd.DataFrame(data, columns=['Problem_Size', 'RHC', 'SA','GA','MIMIC'])
    print("Four Peaks results")
    print(df)

    ax = plt.gca()
    df.plot(kind='line', x='Problem_Size', y='RHC', marker='o', ax=ax)
    df.plot(kind='line', x='Problem_Size', y='SA', marker='o', ax=ax)
    df.plot(kind='line', x='Problem_Size', y='GA', marker='o', ax=ax)
    df.plot(kind='line', x='Problem_Size', y='MIMIC', marker='o', ax=ax)

    plt.xlabel('Problem Size')
    plt.ylabel('Fitness Values')
    plt.title('Randomized Optimization for 4 peaks problem')
    plt.grid(True)

    plt.savefig('4 Peaks Performance.png')

def perform_ro_analysis():
    perform_plot_fourPeaks()
    plt.clf()
    perform_flip_flop()
    plt.clf()
    perform_one_max()
    plt.clf()
