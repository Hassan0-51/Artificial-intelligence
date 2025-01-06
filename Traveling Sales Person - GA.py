import random

def create_distance_matrix(num_cities):
    matrix = [[0 if i == j else random.randint(10, 100) for j in range(num_cities)] for i in range(num_cities)]
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            matrix[j][i] = matrix[i][j] 
    return matrix

def initialize_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

def calculate_distance(tour, distance_matrix):
    distance = 0
    for i in range(len(tour)):
        distance += distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]  
    return distance

def evaluate_fitness(population, distance_matrix):
    fitness_scores = []
    for tour in population:
        distance = calculate_distance(tour, distance_matrix)
        fitness_scores.append(1 / distance) 
    return fitness_scores

def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probs = [fitness / total_fitness for fitness in fitness_scores]
    parent1 = random.choices(population, weights=selection_probs, k=1)[0]
    parent2 = random.choices(population, weights=selection_probs, k=1)[0]
    return parent1, parent2

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size

    child[start:end + 1] = parent1[start:end + 1]

    p2_index = 0
    for i in range(size):
        if child[i] is None:
            while parent2[p2_index] in child:
                p2_index += 1
            child[i] = parent2[p2_index]
    return child

def mutate(tour):
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]  

def genetic_algorithm_tsp(distance_matrix, pop_size, num_generations, mutation_rate=0.1):
    num_cities = len(distance_matrix)
    population = initialize_population(pop_size, num_cities)

    for generation in range(num_generations):
        fitness_scores = evaluate_fitness(population, distance_matrix)

        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population, fitness_scores)

            child = crossover(parent1, parent2)

            if random.random() < mutation_rate:
                mutate(child)

            new_population.append(child)

        population = new_population

        best_fitness = max(fitness_scores)
        best_tour = population[fitness_scores.index(best_fitness)]
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.4f}, Distance = {1 / best_fitness:.2f}")

    best_fitness = max(fitness_scores)
    best_tour = population[fitness_scores.index(best_fitness)]
    return best_tour, 1 / best_fitness

num_cities = 5
pop_size = 10
num_generations = 20

distance_matrix = create_distance_matrix(num_cities)
print("Distance Matrix:")
for row in distance_matrix:
    print(row)

best_tour, best_distance = genetic_algorithm_tsp(distance_matrix, pop_size, num_generations)
print("\nBest Tour:", best_tour)
print("Best Distance:", best_distance)
