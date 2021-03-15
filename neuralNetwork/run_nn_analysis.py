from neuralNetwork.nn_analysis import run_nn_analysis_GD,run_nn_analysis_GA
import pandas as pd
import matplotlib.pyplot as plt

def run_plot_nn():
    bp_accuracy = run_nn_analysis_GD()
    ga_accuracy = run_nn_analysis_GA()

    data = {
        'max_iterations': [100, 200, 300, 400, 500],
        'GD': [bp_accuracy['100_accuracy'], bp_accuracy['200_accuracy'], bp_accuracy['300_accuracy'], bp_accuracy['400_accuracy'], bp_accuracy['500_accuracy']],
        'GA': [ga_accuracy['100_accuracy'], ga_accuracy['200_accuracy'], ga_accuracy['300_accuracy'], ga_accuracy['400_accuracy'], ga_accuracy['500_accuracy']],
        }
    df = pd.DataFrame(data, columns=['max_iterations', 'GD', 'GA'])
    print("Neural Networks GD Vs. GA")
    print(df)

    ax = plt.gca()
    df.plot(kind='line', x='max_iterations', y='GD', marker='o', ax=ax)
    df.plot(kind='line', x='max_iterations', y='GA', marker='o', ax=ax)

    plt.xlabel('Maximum Iterations')
    plt.ylabel('Optimization Algorithms')
    plt.title('Neural Network performance')
    plt.grid(True)

    plt.savefig('NN Performance.png')

if __name__ == '__main__':
    run_plot_nn()