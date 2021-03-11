import mlrose


# Specify a custom function to change the problem to maximization problem
def queens_max(state):
    # Initialize fitness to zero
    fitness = 0
    state_length = len(state)
    # Increment fitness when non attacking queens are found.
    for i in range(state_length - 1):
        for j in range(i + 1, state_length):
            # Check for all attacks and increment only of current queen is safe
            if (state[j] != state[i]) and (state[j] != state[i] + (j - i)) and (state[j] != state[i] - (j - i)):
                fitness += 1
    return fitness


def provide_max_function():
    fitness_cust = mlrose.CustomFitness(queens_max)
    return fitness_cust
