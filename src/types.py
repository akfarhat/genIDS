import random

from deap import base
from deap import creator
from deap import tools

import arff

filename = 'KDDTrain+_20Percent.arff'

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

data = arff.load(open(filename,'rb'))

randomizers = {}

for (attribute, values) in data['attributes']:
    # class is used to tell whether a record is normal or an anomaly
    if attribute == 'class':
        continue

    if isinstance(values, basestring):
        if values == 'REAL':
            randomizers[attribute] = lambda: random.uniform(0, 100000)

    if isinstance(values, list):
        randomizers[attribute] = lambda: random.choice(values)

def random_chromosome():
    chromosome = []

    for (attribute, _) in data['attributes']:
        if attribute == 'class':
            continue

        chromosome.append(randomizers[attribute])

    return chromosome

def empty_chromosome():
    return []

ind_size = len(randomizers.keys())

toolbox = base.Toolbox()
toolbox.register('attr_empty_chromosome', empty_chromosome)
toolbox.register('attr_rand_chromosome', random_chromosome)
toolbox.register("empty_individual", tools.initIterate, creator.Individual, toolbox.attr_empty_chromosome, n=ind_size)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_rand_chromosome, n=ind_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

