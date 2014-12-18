import random
import numbers

from deap import base
from deap import creator
from deap import tools

import arff

# Parameters ##############################

trainset_file = 'KDDTrain_500.arff'
ruleset_file = 'ruleset'
testset_file = 'KDDTest-21_1000.arff'
weight_support = 1 
weight_confidence = 0
NGEN = 50
CXPB = 0.8
MUTPB = 0.1
NPOP = 100
NTEST = 100

###########################################

# Util functions ##########################

def equal_real(a, b):
    return abs(a - b) <= 0.01

##########################################

# Set up GA ##############################

print 'Setting up GA...'

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

data = arff.load(open(trainset_file,'rb'))

randomizers = {}

for (attribute, values) in data['attributes']:
    if (attribute in ('duration','src_bytes','dst_bytes') or
        attribute.startswith('num') or
        attribute.endswith('count')):
        randomizers[attribute] = lambda: random.randint(0,10000)
    elif attribute.endswith('rate'):
        if values == 'REAL':
            randomizers[attribute] = lambda: round(random.random(),2)
    elif isinstance(values, basestring):
        if values == 'REAL':
            randomizers[attribute] = lambda: round(random.uniform(0, 10),2)

    if isinstance(values, list):
        randomizers[attribute] = lambda v=values: random.choice(v)

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
            if isinstance(field, basestring):
                if individual[index] == field:
                    matched_fields += 1.0
            elif isinstance(field, numbers.Number):
                if equal_real(individual[index], field):
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

# Training #########################################

print 'Beginning training...'

pop = toolbox.population(n=NPOP)
fitnesses = toolbox.map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

for g in range(NGEN):
    if g % 10 == 0:
        print 'Generation {}'.format(g)

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

ruleset = pop

f = open(ruleset_file, 'w')
for rule in ruleset:
    f.write(','.join(map(str,rule)))
    f.write('\n')

####################################################

# Testing ##########################################

print 'Beginning testing...'

def equal_connection(a, b):
    for x,y in zip(a,b):
        if isinstance(x, basestring):
            if x != y:
                return False
        elif isinstance(x, numbers.Number):
            if not equal_real(x,y):
                return False

    return True


test_data = arff.load(open(testset_file,'rb'))
tests_to_run = random.sample(test_data['data'], NTEST)

n = len(tests_to_run)
n_attacks = len([test for test in tests_to_run if test[-1] == 'anomaly'])

assert n >= 0
assert n_attacks >= 0

n_positive = 0
n_false = 0
n_miss = 0

for test in tests_to_run:
    found = False

    for rule in ruleset:
        if equal_connection(rule, test):
            if test[-1] == 'anomaly':
                n_positive += 1
                found = True
                break
            else:
                n_false += 1

    if not found:
        if test[-1] == 'anomaly':
            n_miss += 1

print 'Total Tests = {}'.format(n)
print 'Total Attacks = {}'.format(n_attacks)
print 'Num Attacks Detected = {} ({}%)'.format(n_positive, float(n_positive)/float(n_attacks)*100)
print 'Num Attacks Missed = {} ({}%)'.format(n_miss, float(n_miss)/float(n_attacks)*100)
print 'Num False Positives = {}'.format(n_false)
