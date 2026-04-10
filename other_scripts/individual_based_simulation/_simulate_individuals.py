"""This is a very cobbled-together and rough script to demonstrate that the power-gomp can fit other types of model"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_carrying_capacity = 300
mortality_rate = 0.2
YEARS = 100

def main(years = YEARS,
         carrying_capacity = _carrying_capacity,
         plot = False,
         **kwargs):

    assert isinstance(carrying_capacity, int) and carrying_capacity > 0, "carrying_capacity must be a positive integer"

    INITIAL_POPULATION = carrying_capacity
    INITIAL_RESOURCES = carrying_capacity
    RESOURCE_REGROWTH = carrying_capacity
    RESOURCE_CAP = carrying_capacity
    
    P_FORAGE = kwargs.get("P_FORAGE", 1)
    P_REPRODUCE = kwargs.get("P_REPRODUCE", 1)
    P_DEATH = kwargs.get("P_DEATH", 0.15)

    FOOD_PER_FORAGE = kwargs.get("FOOD_PER_FORAGE", 1.0)
    ENERGY_DECAY = kwargs.get("ENERGY_DECAY", 1)

    PEAK_FORAGING_AGE = kwargs.get("PEAK_FORAGING_AGE", 5)
    INITIAL_ENERGY = kwargs.get("INITIAL_ENERGY", 3)
    BREEDING_AGE = kwargs.get("BREEDING_AGE", 3)

    df = pd.DataFrame(columns=["Year", "Population", "Resources"])

    class Resource:
        def __init__(self, amount):
            self.amount = amount

        def regrow(self):
            # self.amount = min(self.amount + np.random.poisson(RESOURCE_REGROWTH), RESOURCE_CAP)
            self.amount = min(self.amount + np.random.poisson(RESOURCE_REGROWTH), RESOURCE_CAP)

    class Individual:
        
        def __init__(self, age=None):
            self.energy = INITIAL_ENERGY
            self.age = np.random.poisson(BREEDING_AGE) if age is None else age
            self.is_breeding = False

        def forage_success_prob(self, resource_amount, resource_cap):
            """Peaks at PEAK_FORAGING_AGE, declines with age; scales with resource availability."""
            age_factor = np.exp(-((self.age - PEAK_FORAGING_AGE) ** 2) / (2 * PEAK_FORAGING_AGE ** 2))
            resource_factor = resource_amount / (resource_cap)
            return age_factor * resource_factor
            # return resource_factor

        def forage(self, resources, resource_cap):
            p = self.forage_success_prob(resources[0].amount, resource_cap)
            if random.random() < p:
                taken = min(FOOD_PER_FORAGE, resources[0].amount)
                resources[0].amount -= taken
                self.energy += taken

        def breeding_success_prob(self):
            """Simple function of energy and age."""
            if self.age < BREEDING_AGE:
                return 0
            # age_factor = np.exp(-((self.age - BREEDING_AGE) ** 2) / (BREEDING_AGE ** 2))
            age_factor = np.exp(-0.3*(self.age - BREEDING_AGE))
            return min(1, age_factor)
        
        def is_alive(self): 
            return self.energy > 0

    def simulate(years=YEARS, initial_population=INITIAL_POPULATION, 
                initial_resources=INITIAL_RESOURCES, 
                resource_regrowth=RESOURCE_REGROWTH, 
                resource_cap=RESOURCE_CAP):
        
        population = [Individual() for _ in range(initial_population)]
        resources = [Resource(initial_resources)]  # list so it's mutable in forage()
        extant = True

        for year in range(1, years + 1):
            resources[0].regrow()

            # Look for food and decay energy
            for ind in population:
                ind.forage(resources, resource_cap)
                ind.energy -= ENERGY_DECAY

            # Reproduction
            random.shuffle(population)
            offspring = []
            for i in range(0, len(population) - 1, 2):
                a, b = population[i], population[i + 1]
                if a.age >= BREEDING_AGE and b.age >= BREEDING_AGE:
                    if random.random() < P_REPRODUCE:
                        offspring.append(Individual(age=0))
        
            population.extend(offspring)

            # Deaths
            population = [ind for ind in population if ind.is_alive() and random.random() > P_DEATH]

            # age
            for ind in population:
                ind.age += 1

            if not population:
                extant = False
                return year, extant

            df.loc[len(df)] = [year, len(population), resources[0].amount]

        return year, extant

    year, extant = simulate()

    if plot:
        df.plot(x="Year", y="Population")
        plt.show()

    return year, extant

if __name__ == "__main__":
    main(carrying_capacity=_carrying_capacity, plot = True, P_DEATH = mortality_rate)  