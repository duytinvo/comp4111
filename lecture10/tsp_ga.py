import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation
from datetime import datetime


# Create two classes: City and Fitness
class City:
    def __init__(self, x, y):
        self.x = x  # latitude
        self.y = y  # longitude
        self.cor = (self.x, self.y)

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

    def getx(self):
        return self.x

    def gety(self):
        return self.y

    def __call__(self):
        return self.x, self.y


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


class GA:
    def __init__(self, population, popSize, eliteSize, mutationRate, breedRate, generations):
        self.population = population
        self.popSize = popSize
        self.eliteSize = eliteSize
        self.mutationRate = mutationRate
        self.breedRate = breedRate
        self.generations = generations
        self.bestPath = self.run()
        pass

    # Create the population
    @staticmethod
    def createRoute(cityList):
        route = random.sample(cityList, len(cityList))
        return route

    @staticmethod
    def initialPopulation(popSize, cityList):
        population = []
        for i in range(0, popSize):
            population.append(GA.createRoute(cityList))
        return population

    # Determine fitness
    @staticmethod
    def rankRoutes(population):
        fitnessResults = {}
        for i in range(0, len(population)):
            fitnessResults[i] = Fitness(population[i]).routeFitness()
        return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

    # Select the mating pool - the roulette wheel selection
    @staticmethod
    def selection(popRanked, eliteSize):
        selectionResults = []
        df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()
        for i in range(0, eliteSize):
            selectionResults.append(popRanked[i][0])

        for i in range(0, len(popRanked) - eliteSize):
            pick = 100 * random.random()
            for i in range(0, len(popRanked)):
                if pick <= df.iat[i, 3]:
                    selectionResults.append(popRanked[i][0])
                    break
        return selectionResults

    @staticmethod
    def matingPool(population, selectionResults):
        matingpool = []
        for i in range(0, len(selectionResults)):
            index = selectionResults[i]
            matingpool.append(population[index])
        return matingpool

    # Breed - aka crossover
    @staticmethod
    def breed(parent1, parent2, breedRate=0.9):
        child = []
        if (random.random() < breedRate):
            childP1 = []
            childP2 = []
            geneA = int(random.random() * len(parent1))
            geneB = int(random.random() * len(parent1))
            while geneB == geneA:
                geneB = int(random.random() * len(parent1))

            startGene = min(geneA, geneB)
            endGene = max(geneA, geneB)
            for i in range(startGene, endGene):
                childP1.append(parent1[i])
            childP2 = [item for item in parent2 if item not in childP1]
            child = childP1 + childP2
        else:
            child = parent1
        return child

    @staticmethod
    def breedPopulation(matingpool, eliteSize, breedRate=0.9):
        children = []
        length = len(matingpool) - eliteSize
        pool = random.sample(matingpool, len(matingpool))

        for i in range(0, eliteSize):
            children.append(matingpool[i])

        for i in range(0, length):
            child = GA.breed(pool[i], pool[len(matingpool) - i - 1], breedRate=breedRate)
            children.append(child)
        return children

    # Mutate - mutation
    @staticmethod
    def mutate(individual, mutationRate):
        for swapped in range(len(individual)):
            if (random.random() < mutationRate):
                swapWith = int(random.random() * len(individual))
                while swapWith == swapped:
                    swapWith = int(random.random() * len(individual))

                city1 = individual[swapped]
                city2 = individual[swapWith]

                individual[swapped] = city2
                individual[swapWith] = city1
        return individual

    @staticmethod
    def mutatePopulation(population, mutationRate):
        mutatedPop = []

        for ind in range(0, len(population)):
            mutatedInd = GA.mutate(population[ind], mutationRate)
            mutatedPop.append(mutatedInd)
        return mutatedPop

    # One generation
    def nextGeneration(self, currentGen):
        popRanked = self.rankRoutes(currentGen)
        selectionResults = self.selection(popRanked, self.eliteSize)
        matingpool = self.matingPool(currentGen, selectionResults)
        children = self.breedPopulation(matingpool, self.eliteSize, self.breedRate)
        nextGeneration = self.mutatePopulation(children, self.mutationRate)
        return nextGeneration

    def run(self):
        pop = self.initialPopulation(self.popSize, self.population)
        bestRoute = self.rankRoutes(pop)[0]
        distance = 1 / bestRoute[1]
        bestPathIndex = bestRoute[0]
        bestPath = pop[bestPathIndex]
        print(f"Distance at generation 0: {distance}")
        progress = []
        progress.append((distance, [path() for path in bestPath]))

        for i in range(0, self.generations):
            pop = self.nextGeneration(pop)
            bestRoute = self.rankRoutes(pop)[0]
            distance = 1 / bestRoute[1]
            bestPathIndex = bestRoute[0]
            bestPath = pop[bestPathIndex]
            progress.append((distance, [path() for path in bestPath]))
            print(f"Distance at generation {i+1}: {distance}")

        plt.plot([prog[0] for prog in progress])
        plt.ylabel('Distance')
        plt.xlabel('Generation')
        plt.show()
        return progress


def plot(progress):
    distances = [p[0] for p in progress]
    paths = [p[1] for p in progress]
    distance = []
    X, Y = (), ()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    line1, = ax1.plot(X, Y, marker='*')
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 200)
    ax1.set_title("Searching Routes")
    ax1.set_xlabel("Latitude")
    ax1.set_ylabel("Longitude")
    line2, = ax2.plot(list(range(len(distance))), distance, marker='^')
    ax2.set_xlim(0, len(distances) + 5)
    ax2.set_ylim(300, max(distances) + 5)
    ax2.set_xlabel("generation")
    ax2.set_ylabel("Distance")
    ax2.set_title("Total Distance of Routes")

    def update(i):
        path = paths[i] + [paths[i][0]]
        X, Y = list(zip(*path))
        line1.set_data(X, Y)
        distance = distances[:i+1]
        line2.set_data(list(range(len(distance))), distance)
        print(i, distance[-1])
        print(path)

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(distances), repeat=True)
    plt.show()


if __name__ == '__main__':
    """
     python tsp_ga.py --seed 12345 --num_city 10 --pop_size 100 --mut_rate 0.01 --bre_rate 0.9 --n_gen 100
     """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--num_city', type=int, default=25, help='Number of cities')
    parser.add_argument('--pop_size', type=int, default=100, help='Population size')
    parser.add_argument('--mut_rate', type=float, default=0.01, help='Mutation rate')
    parser.add_argument('--bre_rate', type=float, default=0.9, help='Breeding rate')
    parser.add_argument('--n_gen', type=int, default=200, help='Number of equal generations before stopping')
    args = parser.parse_args()

    random.seed(args.seed)
    eliteSize = int(args.num_city/2)
    cityList = []
    for i in range(0, args.num_city):
        cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

    tspGA = GA(population=cityList, popSize=args.pop_size, eliteSize=eliteSize,
               mutationRate=args.mut_rate, breedRate=args.bre_rate, generations=args.n_gen)
    progress = tspGA.bestPath
    plot(progress)
