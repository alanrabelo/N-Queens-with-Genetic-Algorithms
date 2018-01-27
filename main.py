import random
from deap import creator, base, tools, algorithms
import numpy as np

SIZE = 4
POPULATION = 2000
NGEN = 150

def sum_diagonals(a, invert=False):

    if invert == True:
        a = np.rot90(a)

    rows, cols = a.shape

    if cols > rows:
        a = a.T
        rows, cols = a.shape
    fill = np.zeros(((cols - 1), cols), dtype=a.dtype)
    stacked = np.vstack((a, fill, a))
    major_stride, minor_stride = stacked.strides
    strides = major_stride, minor_stride * (cols + 1)
    shape = (rows + cols - 1, cols)

    return np.sum(np.array(np.lib.stride_tricks.as_strided(stacked, shape, strides)), axis=1)

def fitness(individual):
    individual = np.array(individual)

    arrayFromIndividual = np.zeros(shape=(SIZE,SIZE))

    for index,value in enumerate(individual):
        arrayFromIndividual[index][value-1] += 1

    sumLines = sum(np.sum(np.array(arrayFromIndividual), axis=0)[np.sum(np.array(arrayFromIndividual), axis=0) > 1])
    sumDiag1 = (sum(value > 1 for value in sum_diagonals(arrayFromIndividual)))
    sumDiag2 = (sum(value > 1 for value in sum_diagonals(arrayFromIndividual, invert=True)))

    return int(sumLines + sumDiag1 + sumDiag2),


# Genetic Algorithm Initialization
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Connecting toolbox with individuals and attributes
toolbox.register("attr_int", random.randint, 1, SIZE)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# AG Main functions
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=SIZE, indpb=0.15)
toolbox.register("select", tools.selTournament, tournsize=3)

# Initial Population setup
population = toolbox.population(n=POPULATION)

# AG Evolutionary Process
for gen in range(NGEN):
    if gen % (NGEN/10) == 0:
        print(int(gen/NGEN*100))
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.25)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# Select best 10 samples from population
top10 = tools.selBest(population, k=10)

for top in top10:
    print(str(fitness(top)) + ' : ' + str(top))


