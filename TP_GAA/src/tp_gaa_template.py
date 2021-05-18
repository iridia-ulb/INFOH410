#!/usr/bin/python3

"""
This implements a genetic algorithm that starts with a base
population of randomly generated strings, iterates over a certain number of
generations while implementing 'natural selection', and prints out the most fit
string.

(c) 2021 Ken Hasselmann

This program subject to the terms of the BSD license listed below.
---
Original code by: Copyright (c) 2011 Colin Drake

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import random
import argparse

#
# Helper functions
# These are used as support, but aren't direct GA-specific functions.
#


def weighted_choice(items):
    """
    Chooses a random element from items, where items is a list of tuples in
    the form (item, weight). weight determines the probability of choosing its
    respective item.
    """
    weight_total = sum((item[1] for item in items))
    n = random.uniform(0, weight_total)
    # print(f"total: {weight_total} {n}")
    for item, weight in items:
        if weight > n:
            # print(f"{weight} {n}")
            return item
        n -= weight
    return items[-1][0]


def random_char():
    """
    Return a random character between ASCII 32 and 126 (i.e. spaces, symbols,
    letters, and digits). All characters returned will be nicely printable.
    """
    return chr(int(random.randrange(32, 126, 1)))


def random_population(size, dna_size):
    """
    Return a list of POP_SIZE individuals, each randomly generated via iterating
    DNA_SIZE times to generate a string of random characters with random_char().
    """
    pop = []
    for _ in range(size):
        dna = ""
        for _ in range(dna_size):
            dna += random_char()
        pop.append(dna)
    return pop


#
# GA functions
# These make up the bulk of the actual GA algorithm.
#


def fitness(dna, optimal):
    # TODO
    return 1


def mutate(dna):
    # TODO
    return dna


def crossover(dna1, dna2, dna_size):
    # TODO
    return (dna1, dna2)


#
# Main driver
# Generate a population and simulate args.gen generations.
#
def main(args):
    # Generate initial population. This will create a list of POP_SIZE strings,
    # each initialized to a sequence of random characters.
    population = random_population(args.pop, len(args.goal))

    # Simulate all of the generations.
    for generation in range(args.gen):
        print(f"Generation {generation}... Random sample: {population[0]}, ", end="")
        weighted_population = []

        # Add individuals and their respective fitness levels to the weighted
        # population list. This will be used to pull out individuals via certain
        # probabilities during the selection phase. Then, reset the population list
        # so we can repopulate it after selection.

        weighted_population = [
            (individual, fitness(individual, args.goal)) for individual in population
        ]
        print(f"Best: {max(weighted_population,key=lambda x:x[1])}")

        # Select two random individuals, based on their fitness probabilites, cross
        # their genes over at a random point, mutate them, and add them back to the
        # population for the next iteration.
        population = []
        # TODO: create new population (fill population list):

    # Display the highest-ranked string after all generations have been iterated
    # over. This will be the closest string to the OPTIMAL string, meaning it
    # will have the smallest fitness value. Finally, exit the program.

    fittest_string = max(population, key=lambda x: fitness(x, args.goal))
    minimum_fitness = fitness(fittest_string, args.goal)

    print(f"Fittest String: {fittest_string}, with score {minimum_fitness}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process args for Genetic alg run")
    parser.add_argument("--gen", help="Nb of generation to run", type=int, default=5000)
    parser.add_argument("--pop", help="Initial population size", type=int, default=20)
    parser.add_argument(
        "--goal",
        help="String to search (optimal sol)",
        type=str,
        default="Hello, World",
    )
    parser.add_argument("--seed", help="Random seed generator", type=int)

    args = parser.parse_args()

    print("parameters:")
    print(args)

    if args.seed:
        random.seed(args.seed)
        # random.seed(args.seed

    main(args)
