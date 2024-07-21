import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import KMeans
import time
import pstats
import cProfile

file_path = '225_assets.txt'
save_filename = '225_assets_UC.xlsx'
# Define parameters
population_size = 400
num_generations = 400
crossover_rate = 1
cannibalism_rate = 0.6
elitism_rate = 0.1
scaling_factor = 0.6 # You may adjust this value
initial_mutation_rate = 0.025
final_mutation_rate = 0.001
step_size_factor = 0.1  # You can adjust this value

# Constraints
cardinality_limit = 10  # Maximum allowed number of assets
budget = 120000  # Budget
lot_size = 1  # Lot size
lambda_value = 0.3 # Replace with your preferred value
stopping_threshold = 10
num_runs = 100
lambda_increment = 0.01
starting_lambda_value = 0


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

# Example usage
returns, cov_matrix = read_data_from_file(file_path)
num_assets = len(returns)

# Function to calculate mean-variance with cardinality constraint
def calculate_mean_variance(portfolios, returns, cov_matrix, lambda_value):
    num_assets = len(returns)
    # Normalize the weights for each portfolio
    normalized_weights = portfolios / np.sum(portfolios, axis=1)[:, None]
    # Calculate portfolio returns
    portfolio_returns = np.dot(normalized_weights, returns)
    # Calculate portfolio risks
    portfolio_risks = np.einsum('ij,ij->i', np.dot(normalized_weights, cov_matrix), normalized_weights)
    # Calculate mean-variances for all portfolios
    mean_variances = lambda_value * portfolio_risks - (1 - lambda_value) * portfolio_returns
    return mean_variances

def de_mutate(population, scaling_factor, mutation_rate):
    num_individuals, num_assets = population.shape
    # Randomly select three distinct individuals for each member in the population
    idx = np.arange(num_individuals)
    r1, r2, r3 = [np.random.choice(idx, size=num_individuals, replace=False) for _ in range(3)]
    # Perform vectorized DE mutation
    trials = population[r1] + scaling_factor * (population[r2] - population[r3])
    # Apply mutation rate in a vectorized manner
    mutation_mask = np.random.rand(num_individuals, num_assets) < mutation_rate
    random_values = np.random.rand(num_individuals, num_assets)
    trials[mutation_mask] = random_values[mutation_mask]
    return trials

# Updated mutate function for vectorized approach
def mutate(population, mutation_rate, scaling_factor):
    mutated_population = de_mutate(population, scaling_factor, mutation_rate)
    mutation_mask = np.random.rand(*population.shape) < mutation_rate
    population[mutation_mask] = mutated_population[mutation_mask]
    return population

# Function to create an initial population with randomly allocated weights
def initialize_population(size, num_assets):
    population = np.random.rand(size, num_assets)
    population /= np.sum(population, axis=1)[:, np.newaxis]  # Normalize weights
    return population

# Function to repair population to satisfy cardinality constraint
def repair_population(population, cardinality_limit):
    """
    Ensure that the portfolios in the population satisfy the cardinality constraint.
    """
    repaired_population = []
    for portfolio in population:
        # Get the indices of the assets with the top weights
        top_assets = np.argpartition(portfolio, -cardinality_limit)[-cardinality_limit:]
        # Create a new portfolio with only these top assets
        new_portfolio = np.zeros_like(portfolio)
        new_portfolio[top_assets] = portfolio[top_assets]
        # Normalize the new portfolio
        new_portfolio /= np.sum(new_portfolio)
        # Add the repaired portfolio to the new population
        repaired_population.append(new_portfolio)
    return np.array(repaired_population)

# Function to select parents based on their fitness (mean-variance)
def select_parents(population, returns, cov_matrix, elitism_rate, lambda_value):
    fitness = calculate_mean_variance(population, returns, cov_matrix, lambda_value)
    sorted_indices = np.argsort(fitness)
    elites = population[sorted_indices[:int(len(population) * elitism_rate)]]
    sorted_fitness = fitness[sorted_indices]
    return elites, sorted_fitness

def build_web(population, best_portfolio_weights, step_size_factor):
    # Ensure best_portfolio_weights is two-dimensional
    if best_portfolio_weights.ndim == 1:
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
    # Generate random pairs of parents from elites
    parent_indices = np.random.choice(len(elites), size=(num_offspring, 2), replace=True)
    parents = elites[parent_indices]

    # Perform crossover in a vectorized manner
    crossover_mask = np.random.rand(num_offspring, num_assets) < crossover_rate
    offspring = np.where(crossover_mask, parents[:, 0, :], parents[:, 1, :])

    return offspring

# Function for cannibalism
def cannibalize(population, elites, elites_count, cannibalism_rate, mutation_rate, scaling_factor):
    num_survivors = int(len(population) * (1 - cannibalism_rate))
    survivors = population[:num_survivors]
    num_offspring = num_survivors // 2 * 2
    num_assets = len(elites[0])

    # Generate offspring using vectorized crossover
    offspring = crossover(elites, num_offspring, num_assets, crossover_rate)

    # Apply vectorized mutation to all offspring at once
    mutated_offspring = mutate(offspring, mutation_rate, scaling_factor)

    # Combine survivors and mutated offspring
    new_population = np.concatenate((survivors, mutated_offspring))

    # Handle remaining population if any
    num_remaining = len(population) - len(new_population)
    if num_remaining > 0:
        remaining_parents_indices = np.random.choice(len(survivors), size=num_remaining, replace=True)
        remaining_parents = survivors[remaining_parents_indices]
        procreated_copies = mutate(remaining_parents, mutation_rate, scaling_factor)
        new_population = np.concatenate((new_population, procreated_copies))

    # Repair the population to satisfy the cardinality constraint
    new_population /= np.sum(new_population, axis=1)[:, np.newaxis]

    return new_population

results_df = pd.DataFrame(columns=['Risk', 'Return', 'Mean Variance', 'Lambda'])
for run in range(num_runs):
    generations_without_improvement = 0  # Reset for each run
    # Set lambda_value for the current run
    lambda_value = (run + 1) * lambda_increment + starting_lambda_value  # Start from 0.3 and increment by 0.05
    # Initialize the population
    population = initialize_population(population_size, num_assets)
    best_mean_variance = float('inf')  # We want to minimize this, so start with infinity
    best_weights = None
    fitness_history = []
    start_time = time.time()
    for generation in range(num_generations):
        # Normalize and ensure non-negative weights at the start of each generation
        population = np.maximum(population, 0)  # Set negative weights to zero
        population /= np.sum(population, axis=1)[:, np.newaxis]

        current_mutation_rate = initial_mutation_rate + (final_mutation_rate - initial_mutation_rate) * (generation / num_generations)

        # Select parents and get sorted fitness values
        elites, sorted_fitness = select_parents(population, returns, cov_matrix, elitism_rate, lambda_value)
        elites_count = len(elites)  # Calculate elites_count based on the length of elites array
        current_best_mean_variance = sorted_fitness[0]  # First element is the lowest mean-variance

        if current_best_mean_variance < best_mean_variance:
            best_weights = elites[0]  # First elite has the best mean variance
            best_mean_variance = current_best_mean_variance
            generations_without_improvement = 0  # Reset the counter as there is an improvement
        else:
            generations_without_improvement += 1  # Increment the counter

        elapsed_generations = generation + 1  # Add 1 as generation starts from 0

        # Web-building phase
        population = build_web(population, best_weights, step_size_factor)

        # Cannibalize and mutate the population
        new_population = cannibalize(population, elites, elitism_rate, cannibalism_rate, current_mutation_rate, scaling_factor)
        population = new_population
        population = mutate(population, current_mutation_rate, scaling_factor)

        # Early termination check
        if generations_without_improvement >= stopping_threshold:
            print("Early stopping triggered after", generation, "generations due to no improvement.")
            break

        # Append the best mean variance to fitness history
        fitness_history.append(best_mean_variance)

    end_time = time.time()
    elapsed_time = end_time-start_time
    iterations_persecond = num_generations/elapsed_time 


    best_portfolio_return = np.dot(returns, best_weights)
    best_portfolio_risk = np.dot(best_weights, np.dot(cov_matrix, best_weights))
        # Print and store the results
    print(f"Run {run + 1} - Best Portfolio Return: {best_portfolio_return}, Best Portfolio Risk: {best_portfolio_risk}, Achieved Mean-Variance: {best_mean_variance}, Lambda: {lambda_value},Elapsed Time: {elapsed_time}")

    # Save the results to the DataFrame
    results_df = results_df._append({
        'Risk': best_portfolio_risk,
        'Return': best_portfolio_return,
        'Mean Variance': best_mean_variance,
        'Lambda': lambda_value,
        'Time': elapsed_time,
        'Elapsed Generations': elapsed_generations  # Store the number of generations
    }, ignore_index=True)

    # Save the results to an Excel file


    results_df.to_excel(save_filename, index=False)

# Calculate mean-variance at the end of the genetic algorithm
print("Run time:",elapsed_time,"seconds")
print("Run Speed:",iterations_persecond,"Generations Per Second")
# Print the results
print(f"Best Portfolio Return: {best_portfolio_return}")
print(f"Best Portfolio Risk: {best_portfolio_risk}")
print(f"Achieved Mean-Variance: {best_mean_variance}")

# Plotting the graphs
plt.plot(fitness_history)
plt.title('Best Mean-Variance Convergence Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Mean-Variance')
plt.grid(True)
plt.show()

# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()

#     main()

#     profiler.disable()
#     profile_filename = 'profile_stats.prof'
#     profiler.dump_stats(profile_filename)

#     with open('profile_output.txt', 'w') as f:
#         stats = pstats.Stats(profile_filename, stream=f)
#         stats.sort_stats('cumtime')  # Sorting by cumulative time
#         stats.print_stats()
#     print("Profiling data saved to 'profile_output.txt'")