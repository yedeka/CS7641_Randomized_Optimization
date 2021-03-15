import numpy as np

import mlrose_hiive

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from yellowbrick.model_selection import LearningCurve

from neuralNetwork.pre_processing import load_cleanse_data

def run_fin_GA():
    prepareddata = load_cleanse_data()
    scaler = MinMaxScaler()

    x_train_scaled = scaler.fit_transform(prepareddata['X_train'])
    x_test_scaled = scaler.transform(prepareddata['X_test'])

    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                          algorithm='genetic_alg', mutation_prob=0.2,
                                          max_iters=100, bias=False, is_classifier=True,
                                          learning_rate=0.001, early_stopping=True,
                                          clip_max=5, max_attempts=50)
    rhcmodel.fit(x_train_scaled, prepareddata['Y_train'])
    y_pred = rhcmodel.predict(x_test_scaled)
    print("Tuned Data Start ------------------------------------------------------------------")
    print(classification_report(prepareddata['Y_test'], y_pred))
    print("Tuned Data End ------------------------------------------------------------------")
    print("Tuned Data Confusion Matrix ------------------------------------------------------------------")
    cm = confusion_matrix(prepareddata['Y_test'], y_pred)
    F1_score = f1_score(prepareddata['Y_test'], y_pred, average="weighted")
    print(F1_score)
    print(cm)

def run_nn_analysis():
    prepareddata = load_cleanse_data()
    scaler = MinMaxScaler()

    x_train_scaled = scaler.fit_transform(prepareddata['X_train'])
    x_test_scaled = scaler.transform(prepareddata['X_test'])

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

def run_nn_analysis_GD():
    prepareddata = load_cleanse_data()
    scaler = MinMaxScaler()

    x_train_scaled = scaler.fit_transform(prepareddata['X_train'])
    x_test_scaled = scaler.transform(prepareddata['X_test'])

    '''iterations = [2, 4, 8, 12, 22, 36, 58, 96, 156, 254, 412, 670, 1090, 1770, 2876, 4670, 7584, 12316, 20000]
    rhc_params = [{'max_iters': iterations}]

    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                    algorithm='gradient_descent',
                                    max_iters=4000, bias=False, is_classifier=True,
                                    learning_rate=[0.001], early_stopping=True,
                                    clip_max=5, max_attempts=50)

    grid = GridSearchCV(rhcmodel, rhc_params, cv=3, scoring='accuracy', verbose=3, refit=True)
    grid.fit(x_train_scaled, prepareddata['Y_train'])

    print("Grid search Data Start ------------------------------------------------------------------")
    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(prepareddata['X_test'])
    print(grid.best_score_)
    print("Grid search Data End ------------------------------------------------------------------") '''


    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                          algorithm='gradient_descent', mutation_prob=0.2,
                                          max_iters=100, bias=False, is_classifier=True,
                                          learning_rate=0.001, early_stopping=True,
                                          clip_max=5, max_attempts=50)
    rhcmodel.fit(x_train_scaled, prepareddata['Y_train'])
    y_pred = rhcmodel.predict(x_test_scaled)
    print(classification_report(prepareddata['Y_test'], y_pred))
    print("Tuned Data End ------------------------------------------------------------------")
    print("Tuned Data Confusion Matrix ------------------------------------------------------------------")
    cm = confusion_matrix(prepareddata['Y_test'], y_pred)
    F1_score_100 = f1_score(prepareddata['Y_test'], y_pred, average="weighted")
    print(F1_score_100)
    print(cm)

    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                          algorithm='gradient_descent', mutation_prob=0.2,
                                          max_iters=200, bias=False, is_classifier=True,
                                          learning_rate=0.001, early_stopping=True,
                                          clip_max=5, max_attempts=50)
    rhcmodel.fit(x_train_scaled, prepareddata['Y_train'])
    y_pred = rhcmodel.predict(x_test_scaled)
    print(classification_report(prepareddata['Y_test'], y_pred))
    print("Tuned Data End ------------------------------------------------------------------")
    print("Tuned Data Confusion Matrix ------------------------------------------------------------------")
    cm = confusion_matrix(prepareddata['Y_test'], y_pred)
    F1_score_200 = f1_score(prepareddata['Y_test'], y_pred, average="weighted")
    print(F1_score_200)
    print(cm)

    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                          algorithm='gradient_descent', mutation_prob=0.2,
                                          max_iters=300, bias=False, is_classifier=True,
                                          learning_rate=0.001, early_stopping=True,
                                          clip_max=5, max_attempts=50)
    rhcmodel.fit(x_train_scaled, prepareddata['Y_train'])
    y_pred = rhcmodel.predict(x_test_scaled)
    print(classification_report(prepareddata['Y_test'], y_pred))
    print("Tuned Data End ------------------------------------------------------------------")
    print("Tuned Data Confusion Matrix ------------------------------------------------------------------")
    cm = confusion_matrix(prepareddata['Y_test'], y_pred)
    F1_score_300 = f1_score(prepareddata['Y_test'], y_pred, average="weighted")
    print(F1_score_300)
    print(cm)

    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                          algorithm='gradient_descent', mutation_prob=0.2,
                                          max_iters=400, bias=False, is_classifier=True,
                                          learning_rate=0.001, early_stopping=True,
                                          clip_max=5, max_attempts=50)
    rhcmodel.fit(x_train_scaled, prepareddata['Y_train'])
    y_pred = rhcmodel.predict(x_test_scaled)
    print(classification_report(prepareddata['Y_test'], y_pred))
    print("Tuned Data End ------------------------------------------------------------------")
    print("Tuned Data Confusion Matrix ------------------------------------------------------------------")
    cm = confusion_matrix(prepareddata['Y_test'], y_pred)
    F1_score_400 = f1_score(prepareddata['Y_test'], y_pred, average="weighted")
    print(F1_score_400)
    print(cm)

    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                          algorithm='gradient_descent', mutation_prob=0.2,
                                          max_iters=500, bias=False, is_classifier=True,
                                          learning_rate=0.001, early_stopping=True,
                                          clip_max=5, max_attempts=50)
    rhcmodel.fit(x_train_scaled, prepareddata['Y_train'])
    y_pred = rhcmodel.predict(x_test_scaled)
    print(classification_report(prepareddata['Y_test'], y_pred))
    print("Tuned Data End ------------------------------------------------------------------")
    print("Tuned Data Confusion Matrix ------------------------------------------------------------------")
    cm = confusion_matrix(prepareddata['Y_test'], y_pred)
    F1_score_500 = f1_score(prepareddata['Y_test'], y_pred, average="weighted")
    print(F1_score_500)
    print(cm)

    return {
        '100_accuracy': F1_score_100,
        '200_accuracy':F1_score_200,
        '300_accuracy':F1_score_300,
        '400_accuracy':F1_score_400,
        '500_accuracy':F1_score_500
    }

def run_nn_analysis_GA():
    prepareddata = load_cleanse_data()
    scaler = MinMaxScaler()

    x_train_scaled = scaler.fit_transform(prepareddata['X_train'])
    x_test_scaled = scaler.transform(prepareddata['X_test'])

    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                          algorithm='genetic_alg', mutation_prob=0.2,
                                          max_iters=100, bias=False, is_classifier=True,
                                          learning_rate=0.001, early_stopping=True,
                                          clip_max=5, max_attempts=50)
    rhcmodel.fit(x_train_scaled, prepareddata['Y_train'])
    y_pred = rhcmodel.predict(x_test_scaled)
    print("Tuned Data Start ------------------------------------------------------------------")
    print(classification_report(prepareddata['Y_test'], y_pred))
    print("Tuned Data End ------------------------------------------------------------------")
    print("Tuned Data Confusion Matrix ------------------------------------------------------------------")
    cm = confusion_matrix(prepareddata['Y_test'], y_pred)
    F1_score_100 = f1_score(prepareddata['Y_test'], y_pred, average="weighted")
    print(F1_score_100)
    print(cm)

    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                          algorithm='genetic_alg', mutation_prob=0.2,
                                          max_iters=200, bias=False, is_classifier=True,
                                          learning_rate=0.001, early_stopping=True,
                                          clip_max=5, max_attempts=50)
    rhcmodel.fit(x_train_scaled, prepareddata['Y_train'])
    y_pred = rhcmodel.predict(x_test_scaled)
    print("Tuned Data Start ------------------------------------------------------------------")
    print(classification_report(prepareddata['Y_test'], y_pred))
    print("Tuned Data End ------------------------------------------------------------------")
    print("Tuned Data Confusion Matrix ------------------------------------------------------------------")
    cm = confusion_matrix(prepareddata['Y_test'], y_pred)
    F1_score_200 = f1_score(prepareddata['Y_test'], y_pred, average="weighted")
    print(F1_score_200)
    print(cm)

    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                          algorithm='genetic_alg', mutation_prob=0.2,
                                          max_iters=300, bias=False, is_classifier=True,
                                          learning_rate=0.001, early_stopping=True,
                                          clip_max=5, max_attempts=50)
    rhcmodel.fit(x_train_scaled, prepareddata['Y_train'])
    y_pred = rhcmodel.predict(x_test_scaled)
    print("Tuned Data Start ------------------------------------------------------------------")
    print(classification_report(prepareddata['Y_test'], y_pred))
    print("Tuned Data End ------------------------------------------------------------------")
    print("Tuned Data Confusion Matrix ------------------------------------------------------------------")
    cm = confusion_matrix(prepareddata['Y_test'], y_pred)
    F1_score_300 = f1_score(prepareddata['Y_test'], y_pred, average="weighted")
    print(F1_score_300)
    print(cm)

    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                          algorithm='genetic_alg', mutation_prob=0.2,
                                          max_iters=400, bias=False, is_classifier=True,
                                          learning_rate=0.001, early_stopping=True,
                                          clip_max=5, max_attempts=50)
    rhcmodel.fit(x_train_scaled, prepareddata['Y_train'])
    y_pred = rhcmodel.predict(x_test_scaled)
    print("Tuned Data Start ------------------------------------------------------------------")
    print(classification_report(prepareddata['Y_test'], y_pred))
    print("Tuned Data End ------------------------------------------------------------------")
    print("Tuned Data Confusion Matrix ------------------------------------------------------------------")
    cm = confusion_matrix(prepareddata['Y_test'], y_pred)
    F1_score_400 = f1_score(prepareddata['Y_test'], y_pred, average="weighted")
    print(F1_score_400)
    print(cm)

    rhcmodel = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], activation='sigmoid',
                                          algorithm='genetic_alg', mutation_prob=0.2,
                                          max_iters=500, bias=False, is_classifier=True,
                                          learning_rate=0.001, early_stopping=True,
                                          clip_max=5, max_attempts=50)
    rhcmodel.fit(x_train_scaled, prepareddata['Y_train'])
    y_pred = rhcmodel.predict(x_test_scaled)
    print("Tuned Data Start ------------------------------------------------------------------")
    print(classification_report(prepareddata['Y_test'], y_pred))
    print("Tuned Data End ------------------------------------------------------------------")
    print("Tuned Data Confusion Matrix ------------------------------------------------------------------")
    cm = confusion_matrix(prepareddata['Y_test'], y_pred)
    F1_score_500 = f1_score(prepareddata['Y_test'], y_pred, average="weighted")
    print(F1_score_500)
    print(cm)

    return {
        '100_accuracy': F1_score_100,
        '200_accuracy':F1_score_200,
        '300_accuracy':F1_score_300,
        '400_accuracy':F1_score_400,
        '500_accuracy':F1_score_500
    }

