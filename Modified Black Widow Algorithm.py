import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    num_assets = int(lines[0].strip())
    returns = []
    std_devs = []
    # Read returns and standard deviations
    for i in range(1, num_assets + 1):
        line = lines[i].strip().split()
        returns.append(float(line[0]))
        std_devs.append(float(line[1]))
    # Read correlation matrix
    correlation_matrix = np.zeros((num_assets, num_assets))
    for i in range(num_assets + 1, len(lines)):
        line = lines[i].strip().split()
        x, y, corr = int(line[0]), int(line[1]), float(line[2])
        correlation_matrix[x-1, y-1] = corr
        correlation_matrix[y-1, x-1] = corr  # Symmetric
    # Set diagonal to 1 (correlation with itself)
    np.fill_diagonal(correlation_matrix, 1)
    # Convert correlation matrix to covariance matrix
    std_devs_matrix = np.outer(std_devs, std_devs)
    cov_matrix = correlation_matrix * std_devs_matrix
    return np.array(returns), cov_matrix

# file Reading usage
file_path = '225_assets.txt'
save_filename = '225_assets_Constrained.xlsx'
returns, cov_matrix = read_data_from_file(file_path)
num_assets = len(returns)

num_runs = 51
lambda_increment = 0.02
starting_lambda_value = -.02

# Define parameters
population_size = 200
num_generations = 3000
crossover_rate = 1
cannibalism_rate = 0.9
elitism_rate = 0.1
scaling_factor = 0.9 
step_size_factor = 0.2
current_mutation_rate = 0.0091

# Constraints
cardinality_limit = 10  # Maximum allowed number of assets
lambda_value = 0.3 # Replace with your preferred value
stopping_threshold = 10
max_weight = 1
min_weight = 0.01 # Maximum allowed weight for any asset in a portfolio
penalty_factor = 1
penalty_factor_max = 1
penalty_factor_min = 1
min_weight_penalty_factor = 1

def combined_penalty(portfolios, max_weight, min_weight, penalty_factor_max, penalty_factor_min, cardinality_limit):
    # Vectorized penalty for exceeding max weight
    penalty_max = np.maximum(portfolios - max_weight, 0) * penalty_factor_max   
    # Vectorized penalty for being below min weight, applied only to non-zero weights
    penalty_min = np.maximum(min_weight - portfolios, 0) * penalty_factor_min
    penalty_min *= (portfolios > 0)
    # Vectorized cardinality penalty
    num_assets_in_portfolio = np.sum(portfolios > 0, axis=1)
    cardinality_penalty = np.maximum(cardinality_limit - num_assets_in_portfolio, 0) * penalty_factor_min
    # Combine all penalties
    total_penalty = np.sum(penalty_max, axis=1) + np.sum(penalty_min, axis=1) + cardinality_penalty
    return total_penalty

def calculate_mean_variance(portfolios, returns, cov_matrix, lambda_value, max_weight, min_weight, penalty_factor_max, penalty_factor_min, cardinality_limit):
    portfolio_returns = np.dot(portfolios, returns)
    portfolio_risks = np.einsum('ij,ij->i', np.dot(portfolios, cov_matrix), portfolios)
    basic_mean_variances = lambda_value * portfolio_risks - (1 - lambda_value) * portfolio_returns
    total_penalty = combined_penalty(portfolios, max_weight, min_weight, penalty_factor_max, penalty_factor_min, cardinality_limit)
    penalized_mean_variances = basic_mean_variances + total_penalty
    return penalized_mean_variances

def de_mutate(population, scaling_factor, mutation_rate):
    num_individuals, num_assets = population.shape
    idx = np.arange(num_individuals)
    r1, r2, r3 = [np.random.choice(idx, size=num_individuals, replace=False) for _ in range(3)]
    trials = population[r1] + scaling_factor * (population[r2] - population[r3])
    mutation_mask = np.random.rand(num_individuals, num_assets) < mutation_rate
    random_values = np.random.rand(num_individuals, num_assets)
    trials[mutation_mask] = random_values[mutation_mask]
    return trials

# Function to create an initial population with randomly allocated weights
def initialize_population(size, num_assets, cardinality_limit):
    population = np.zeros((size, num_assets))
    for i in range(size):
        selected_indices = np.random.choice(num_assets, cardinality_limit, replace=False)
        population[i, selected_indices] = np.random.rand(cardinality_limit)
        population[i, :num_assets] /= np.sum(population[i, :num_assets])  # Normalize weights
    return population

# Function to repair population to satisfy cardinality constraint
def repair_population(population, cardinality_limit):
    top_assets_indices = np.argpartition(population, -cardinality_limit, axis=1)[:, -cardinality_limit:]
    repaired_population = np.zeros_like(population)
    rows = np.indices(top_assets_indices.shape)
    repaired_population[rows, top_assets_indices] = population[rows, top_assets_indices]
    return repaired_population

# Function to select parents based on their fitness (mean-variance)
def select_parents(population, returns, cov_matrix, elitism_rate, lambda_value, max_weight):
    mean_variances = calculate_mean_variance(population, returns, cov_matrix, lambda_value,max_weight,min_weight,penalty_factor_max, penalty_factor_min,cardinality_limit)
    sorted_indices = np.argsort(mean_variances)
    elites = population[sorted_indices[:int(len(population) * elitism_rate)]]
    sorted_fitness = mean_variances[sorted_indices]
    return elites, sorted_fitness

def build_web(population, best_portfolio_weights, step_size_factor):
    # Ensure best_portfolio_weights is two-dimensional
    best_portfolio_weights = best_portfolio_weights[np.newaxis, :]
    # Repeat the best portfolios to match the population size
    num_portfolios = len(population)
    repeat_times = num_portfolios // len(best_portfolio_weights)
    repeated_best_portfolios = np.repeat(best_portfolio_weights, repeat_times, axis=0)
    # Calculate the direction and step size
    direction = repeated_best_portfolios - population
    step_size = step_size_factor * direction
    # Update the population towards the best portfolio
    adjusted_population = population + step_size
    return adjusted_population

# Function for crossover (Uniform-point crossover)
def crossover(elites, num_offspring, num_assets, crossover_rate):
    parent_indices = np.random.choice(len(elites), size=(num_offspring, 2), replace=True)
    parents = elites[parent_indices]
    crossover_mask = np.random.rand(num_offspring, num_assets) < crossover_rate
    offspring = np.where(crossover_mask, parents[:, 0, :], parents[:, 1, :])
    return offspring

# Function for cannibalism
def evolve_population(population, elites, cannibalism_rate, mutation_rate, scaling_factor, cardinality_limit, crossover_rate):
    num_survivors = int(len(population) * (1 - cannibalism_rate))
    num_offspring = (len(population) - num_survivors) // 2 * 2  # Ensure even number for crossover
    num_assets = len(elites[0])
    # Perform crossover among elites to produce offspring
    parent_indices = np.random.choice(len(elites), size=(num_offspring, 2), replace=True)
    parents = elites[parent_indices]
    crossover_mask = np.random.rand(num_offspring, num_assets) < crossover_rate
    offspring = np.where(crossover_mask, parents[:, 0, :], parents[:, 1, :])
    # Mutate offspring and remaining population
    mutated_offspring = de_mutate(offspring, scaling_factor, mutation_rate)
    remaining_parents_indices = np.random.choice(num_survivors, size=(len(population) - num_survivors - num_offspring), replace=True)
    remaining_parents = population[:num_survivors][remaining_parents_indices]
    mutated_remaining = remaining_parents
    # Combine survivors, mutated offspring, and mutated remaining population
    new_population = np.concatenate([population[:num_survivors], mutated_offspring, mutated_remaining])
    # Normalize and repair new population
    new_population = repair_population(new_population, cardinality_limit)

    return new_population

results_df = pd.DataFrame(columns=['Risk', 'Return', 'Mean Variance', 'Lambda'])
for run in range(num_runs):
    generations_without_improvement = 0  # Reset for each run
    # Set lambda_value for the current run
    lambda_value = (run + 1) * lambda_increment + starting_lambda_value
    population = initialize_population(population_size, num_assets, cardinality_limit)
    best_mean_variance = float('inf')  # We want to minimize this, so start with infinity
    best_weights = None
    penalty_history = []
    fitness_history = []
    start_time = time.time()
    for generation in range(num_generations):
        population /= np.sum(population, axis=1)[:, np.newaxis]
        elites, sorted_fitness = select_parents(population, returns, cov_matrix, elitism_rate, lambda_value, max_weight)
        current_best_mean_variance = sorted_fitness[0]
        if current_best_mean_variance < best_mean_variance:
            best_weights = elites[0]
            best_mean_variance = current_best_mean_variance
        population = build_web(population, best_weights, step_size_factor)
        population = evolve_population(population, elites, cannibalism_rate, current_mutation_rate, scaling_factor, cardinality_limit, crossover_rate)
        current_generation = generation
        fitness_history.append(best_mean_variance)


    end_time = time.time()
    elapsed_time = end_time-start_time
    iterations_persecond = current_generation/elapsed_time 
    best_portfolio_return = np.dot(returns, best_weights)
    best_portfolio_risk = np.dot(best_weights, np.dot(cov_matrix, best_weights))
    end_time = time.time()
    elapsed_time = end_time-start_time
    iterations_persecond = num_generations/elapsed_time 
    best_portfolio_return = np.dot(returns, best_weights)
    best_portfolio_risk = np.dot(best_weights, np.dot(cov_matrix, best_weights))
    print(f"Run {run + 1} - Best Portfolio Return: {best_portfolio_return}, Best Portfolio Risk: {best_portfolio_risk}, Achieved Mean-Variance: {best_mean_variance}, Lambda: {lambda_value},Elapsed Time: {elapsed_time}")

    # Save the results to the DataFrame
    results_df = results_df._append({
        'Risk': best_portfolio_risk,
        'Return': best_portfolio_return,
        'Mean Variance': best_mean_variance,
        'Lambda': lambda_value,
        'Time': elapsed_time,
        'Elapsed Generations': generation  # Store the number of generations
    }, ignore_index=True)
    results_df.to_excel(save_filename, index=False)

print("Run time:",elapsed_time,"seconds")
print("Run Speed:",iterations_persecond,"Generations Per Second")
# Print the results
print(f"Best Portfolio Return: {best_portfolio_return}")
print(f"Best Portfolio Risk: {best_portfolio_risk}")
print(f"Achieved Mean-Variance: {best_mean_variance}")
print(best_weights)
# Calculate mean-variance at the end of the genetic algorithm
print("Run time:",elapsed_time,"seconds")
print("Run Speed:",iterations_persecond,"Generations Per Second")
# Print the results
print(f"Best Portfolio Return: {best_portfolio_return}")
print(f"Best Portfolio Risk: {best_portfolio_risk}")
print(f"Achieved Mean-Variance: {best_mean_variance}")

# Plotting the graphs
plt.figure(figsize=(10, 5))

# Plot fitness history
plt.subplot(1, 2, 1)
plt.plot(fitness_history)
plt.title('Best Mean-Variance Convergence Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Mean-Variance')
plt.grid(True)

# Plot penalty history
plt.subplot(1, 2, 2)
plt.plot(penalty_history)
plt.title('Mean Variance Penalty Over Generations')
plt.xlabel('Generation')
plt.ylabel('Penalty')
plt.grid(True)

plt.tight_layout()
plt.show()
