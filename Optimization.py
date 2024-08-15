import random
from deap import base, creator, tools, algorithms
import pickle
import pandas as pd
import numpy as np

def optimize(csvPath):
    data = pd.read_csv(csvPath)
    objectvalue = []
    decisionva = []
    for p in range(len(data)):
        data = data[p: p + 1]
        # Define the problem
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        def evaluate(individual):
            with open('model_Wear_RF.pkl', 'rb') as f:
                model_Wear = pickle.load(f)
            with open('model_SE_RF.pkl', 'rb') as t:
                model_SE = pickle.load(t)
            input_keys = ['Burial depth', 'Poisson ratio', 'Soil density', 'Coefficient of earth pressure', 'Radius',
                          'Water content', 'Torque',
                          'Hydraulic conductivity']
            thrust, rotational, penetration, advspeed = individual[0], individual[1], individual[2], individual[3]
            input_data = data[input_keys]
            input_data['Thrust'] = thrust
            input_data['Rotational'] = rotational
            input_data['Penetration'] = penetration
            input_data['Advanced speed'] = advspeed
            input_data_SE = data[input_keys + ['Wear']]
            input_data_SE['Thrust'] = thrust
            input_data_SE['Rotational'] = rotational
            input_data_SE['Penetration'] = penetration
            input_data_SE['Advanced speed'] = advspeed
            wear = model_Wear.predict(input_data.values)
            specific_energy = model_SE.predict(input_data_SE.values)
            return float(wear), float(specific_energy)

        # Set up the optimization algorithm
        toolbox = base.Toolbox()

        # Define the decision variables
        toolbox.register("thrust", random.uniform, 5000, 20000)
        toolbox.register("rotationspeed", random.uniform, 0.01, 1.8)
        toolbox.register("penetration", random.uniform, 0.01, 60)
        toolbox.register("advspeed", random.uniform, 0.01, 15)

        # Define the mutation and crossover operators
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.thrust, toolbox.rotationspeed, toolbox.penetration, toolbox.advspeed))
        individual = toolbox.individual()
        toolbox.register("population", tools.initCycle, list, toolbox.individual)
        lower_bound = [5000, 0.01, 0.01, 0.01]
        upper_bound = [20000, 1.8, 60, 15]

        toolbox.register("evaluate", evaluate)

        toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=lower_bound, up=upper_bound)
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=lower_bound, up=upper_bound,
                         indpb=1.0 / len(individual))
        toolbox.register("select", tools.selNSGA2)
        population = [toolbox.individual() for i in range(50)]  # population_size

        # Run the optimization algorithm
        for generation in range(10):  # num_generations
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.7,
                                          mutpb=0.7)  # crossover_probability mutation_probability
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=50)

        # Analyze the results
        for ind in population:
            objectvalue.append(ind.fitness.values)
            decisionva.append(ind)

    outmulti = pd.DataFrame(data={'Objective': objectvalue,
                                  'decision': decisionva})
    outmulti.to_csv('..\optimizeresult.csv')
