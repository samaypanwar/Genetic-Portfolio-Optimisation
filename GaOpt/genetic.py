from numpy import random

class GeneticAlgorithm:

    """This Class handles all the optimisation for the Genetic Algorithm.
    """

    def __init__(self, fitnessFunction, dataframe, longOnly = False):

        # Number of individuals in our population
        self.populationSize = 1000
        # How much information can each parameter take
        self.nBits = 16
        # Number of populations to make
        self.nGenerations = 5
        # Chance of a child having atleast one different gene when compared to its parents
        self.crossoverRate = 0.98
        # Bounds for our characteristics (weights in our case)
        self.maxNumber = 1.0
        self.bounds = [
            [-self.maxNumber, self.maxNumber] for ticker in dataframe.columns
        ]
        self.mutatationRate = 1.0 / (float(len(self.bounds)))

        # Function that we want to optimise
        self.fitnessFunction = lambda individual: fitnessFunction(dataframe, individual)

        # Number of individuals in a cluster
        self.randomParentSample = 25

        # Population ranges between all the pre-set bounds
        # Each individual in our population is a nBits x len(self.bounds) matrix where one parameter is encoded
        # into nBits and the same is done for the number of tickers in our portfolio dataframe

        self.population = [
            self.create_individual(longOnly)
            for _ in range(self.populationSize)
        ]

    def evolution(self, verbose = False):

        # Take the best individual with the best parameters to be the first one in our population initially
        bestChild = 0

        # Take the best current score to be the value of the fitness function with the first individual
        bestScore = self.fitnessFunction(self.population[0])

        for generation in range(self.nGenerations):

            if verbose:
                print("\n---Generation {} has begun----\n".format(generation + 1))

            # Create a list of decoded params so the matrix shape changes from nBits x tickers into 1 x tickers for
            # each individual in our population
            # decoded = [self.decode(candidate) for candidate in self.population]

            # Score each individual set of parameters on our fitness function for the entire current population
            scores = [self.fitnessFunction(candidate) for candidate in self.population]

            # The size of the score array and the population should be the same
            assert len(scores) == len(self.population)

            # A parent is chosen from a random possible set of parents which has the best
            # score amongst the random sample
            allParents = [
                self.select_parent(scores) for _ in range(self.populationSize)
            ]

            # Each generation is initialised such that it has no children initiallly
            children = []

            # For each individual (set of weights) in our population
            for individual in range(self.populationSize):

                # Find the best parameters/genes that maximise our fitnessFunction
                if scores[individual] > bestScore:
                    bestScore, bestChild = scores[individual].copy(), self.population[individual].copy()

                    if verbose:
                        print(f"-- generation {generation+1}, New best fitness score: {bestScore}")

            for parentIndex in range(0, self.populationSize - 1, 2):

                # Take two adjacent parents and make their children
                # Can also use a random sample of two people from our allParents
                parents = allParents[parentIndex : parentIndex + 2]

                for child in self.crossover(parents):

                    # Induce a mutation into the child genes and add it to
                    # the current generation of children
                    self.mutation(child)
                    children.append(child)

            # Previous generation of population is replaced by the children
            self.population = children

            for individual in range(len(self.population)):
                if sum(self.population[individual]) != 1:
                    self.population[individual] = self.population[individual] / sum(self.population[individual])

        return bestChild

    def mutation(self, individual):

        if random.rand() < self.mutatationRate:
            for gene in range(len(individual)):
                individual[gene] = 1 - individual[gene]

    def crossover(self, parents):

        # Make both kids exact copies of their parents
        childOne, childTwo = parents.copy()[0], parents.copy()[1]

        # There's a chance that a kid may perfectly resemble one of the parents
        if random.rand() < self.crossoverRate:

            # For each of the children, they can take up genes from either one of their parents
            for child in [childOne, childTwo]:

                # Each gene has equal probabilty of coming from either parent
                parentOneGene = 0.50

                # For each possible gene(bit) in a child
                for gene in range(len(childOne)):

                    if random.rand() < parentOneGene:
                        child[gene] = parents[0][gene]

                    else:
                        child[gene] = parents[1][gene]

        return [childOne, childTwo]

    def select_parent(self, scores):

        # Random sample size to choose a parent form
        access = self.randomParentSample
        selected_index = random.randint(0, self.populationSize)

        # A array of random parents is chosen and the one with the best score is returned
        for candidate in random.randint(0, self.populationSize, access - 1):

            if scores[candidate] > scores[selected_index]:
                selected_index = candidate

        # Returns the individual with the best score amongst the chosen sample
        return self.population[selected_index]

    def create_individual(self, longOnly):

        minVal = -1
        maxVal = 1

        individual = random.rand(len(self.bounds))
        individual = individual / sum(individual)

        if not longOnly:
            individual = minVal + (individual * (maxVal - minVal))

        return individual
