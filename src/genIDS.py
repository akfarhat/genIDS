import random

from deap import base
from deap import creator
from deap import tools

import arff

# Parameters ##############################

trainset_file = 'KDDTrain_5Percent.arff'
ruleset_file = 'ruleset'
weight_support = 1 
weight_confidence = 0
NGEN = 10
CXPB = 0.8
MUTPB = 0.1
NPOP = 100

###########################################

# Set up GA ##############################

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

data = arff.load(open(trainset_file,'rb'))

randomizers = {}

for (attribute, values) in data['attributes']:
    if isinstance(values, basestring):
        if values == 'REAL':
            randomizers[attribute] = lambda: random.uniform(0, 100000)

    if isinstance(values, list):
        randomizers[attribute] = lambda: random.choice(values)

def random_value(i):
    return randomizers[data['attributes'][i][0]]()

def random_chromosome():
    chromosome = []

    for (attribute, _) in data['attributes']:
        chromosome.append(randomizers[attribute]())

    return chromosome

def empty_chromosome():
    return []

ind_size = len(randomizers.keys())

toolbox = base.Toolbox()
toolbox.register('attr_empty_chromosome', empty_chromosome)
toolbox.register('attr_rand_chromosome', random_chromosome)
toolbox.register("empty_individual", tools.initIterate, creator.Individual, toolbox.attr_empty_chromosome)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_rand_chromosome)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# fitness function to evaluate individual
def evaluate(individual):
    N = float(len(data['data']))
    A = 0.0
    AB = 0.0
    w1 = weight_support
    w2 = weight_confidence

    for record in data['data']:
        matched_fields = 0.0

        for index, field in enumerate(record, start=0):
            if (individual[index] == field):
                matched_fields += 1.0
            if index == ind_size-2 and matched_fields == ind_size-1: 
                A += 1
            if index == ind_size-1 and matched_fields == ind_size:
                AB += 1
    
    support = AB/N
    
    if A > 0:
        confidence = AB/A
    else:
        confidence = 0.0
    
    fitness = w1 * support + w2 * confidence

    return fitness,

def mutate(individual, mut_threshold):
    for i, attribute in enumerate(individual):
        if random.random() < mut_threshold:
            individual[i] = random_value(i)

toolbox.register('evaluate', evaluate)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('crossover', tools.cxTwoPoint)
toolbox.register('mutate', mutate, mut_threshold=MUTPB)

####################################################

# Run Evolution ###################################

pop = toolbox.population(n=NPOP)
fitnesses = toolbox.map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

for g in range(NGEN):
    offspring = toolbox.select(pop, len(pop))
    offspring = map(toolbox.clone, offspring)

    for child1, child2, in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.crossover(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring

f = open(ruleset_file, 'w')
for individual in pop:
    f.write(','.join(individual))
    f.write('\n')

########################################################
