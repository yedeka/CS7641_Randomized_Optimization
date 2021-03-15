from neuralNetwork.nn_analysis import run_nn_analysis_GD,run_nn_analysis_GA

def run_plot_nn():
    bp_accuracy = run_nn_analysis_GD()
    GA_accuracy = run_nn_analysis_GA()


if __name__ == '__main__':
    run_plot_nn()