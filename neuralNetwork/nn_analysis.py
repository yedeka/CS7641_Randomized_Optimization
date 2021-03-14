import numpy as np

import mlrose_hiive

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from neuralNetwork.pre_processing import load_cleanse_data

def run_nn_analysis():
    prepareddata = load_cleanse_data()
    scaler = MinMaxScaler()

    x_train_scaled = scaler.fit_transform(prepareddata['X_train'])
    x_test_scaled = scaler.transform(prepareddata['X_test'])

    # iterations = np.logspace(0, 3, 30).astype(int) * 2
    iterations = [2, 4, 8, 12, 22, 36, 58, 96, 156, 254, 412, 670, 1090, 1770, 2876, 4670, 7584, 12316, 20000]
    rhc_params = [{'max_iters': iterations,
                   'mutation_prob': np.arange(.001, .1, .01).astype(float)}]

    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                    algorithm='genetic_alg',
                                    max_iters=4000, bias=False, is_classifier=True,
                                    learning_rate=0.001, early_stopping=True,
                                    clip_max=5, max_attempts=50)

    grid = GridSearchCV(rhcmodel, rhc_params, cv=3, scoring='accuracy', verbose=3, refit=True)
    grid.fit(x_train_scaled, prepareddata['Y_train'])

    print("Grid search Data Start ------------------------------------------------------------------")
    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(prepareddata['X_test'])
    # print classification report
    print(grid.best_score_)
    print("Grid search Data End ------------------------------------------------------------------")

def run_nn_analysis_SA():
    prepareddata = load_cleanse_data()
    scaler = MinMaxScaler()

    x_train_scaled = scaler.fit_transform(prepareddata['X_train'])
    x_test_scaled = scaler.transform(prepareddata['X_test'])

    iterations = [2, 4, 8, 12, 22, 36, 58, 96, 156, 254, 412, 670, 1090, 1770, 2876, 4670, 7584, 12316, 20000]
    rhc_params = [{'max_iters': iterations}]

    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                    algorithm='simulated_annealing',
                                    max_iters=4000, bias=False, is_classifier=True,
                                    learning_rate=0.001, early_stopping=True, schedule=mlrose_hiive.GeomDecay(10, .99, .1),
                                    clip_max=5, max_attempts=50)

    grid = GridSearchCV(rhcmodel, rhc_params, cv=3, scoring='accuracy', verbose=3, refit=True)
    grid.fit(x_train_scaled, prepareddata['Y_train'])

    print("Grid search Data Start ------------------------------------------------------------------")
    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(prepareddata['X_test'])
    # print classification report
    print(grid.best_score_)
    print("Grid search Data End ------------------------------------------------------------------")

def run_nn_analysis_RHC():
    prepareddata = load_cleanse_data()
    scaler = MinMaxScaler()

    x_train_scaled = scaler.fit_transform(prepareddata['X_train'])
    x_test_scaled = scaler.transform(prepareddata['X_test'])

    iterations = [2, 4, 8, 12, 22, 36, 58, 96, 156, 254, 412, 670, 1090, 1770, 2876, 4670, 7584, 12316, 20000]
    rhc_params = [{'max_iters': iterations}]

    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                    max_iters=4000, bias=False, is_classifier=True,
                                    learning_rate=0.001, early_stopping=True,
                                    clip_max=5, max_attempts=50)

    grid = GridSearchCV(rhcmodel, rhc_params, cv=3, scoring='accuracy', verbose=3, refit=True)
    grid.fit(x_train_scaled, prepareddata['Y_train'])

    print("Grid search Data Start ------------------------------------------------------------------")
    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(prepareddata['X_test'])
    # print classification report
    print(grid.best_score_)
    print("Grid search Data End ------------------------------------------------------------------")
