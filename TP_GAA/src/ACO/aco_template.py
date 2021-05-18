#!/usr/bin/python3

# Created on: May 2021
# Author: Ken Hasselmann <ken.hasselmann * ulb.be>

import argparse
import numpy as np
import parseTSP

# For visualization
# import animation


class Ant:
    def __init__(self, tsp, probability, rng):
        self.tsp = tsp
        self.probability = probability
        self.rng = rng
        self.tour = []
        self.tour_length = 0

    def __str__(self):
        return f"Ant, tour_l: {self.tour_length}"

    def search(self):
        self.tour = []
        self.tour_length = 0
        # select first city
        self.tour.append(rng.integers(0, len(self.tsp) - 1))
        for _ in range(len(self.tsp)):
            self.tour.append(self.get_next_city())

    def get_next_city(self):
        # TODO: Implement here
        next_city = 0
        return next_city

    def compute_tour_length(self):
        if self.tour_length != 0:
            return self.tour_length

        self.tour_length = sum(
            [self.tsp.dist(a, b) for a, b in zip(self.tour, self.tour[1:])]
        )
        self.tour_length += self.tsp.dist(self.tour[-1], self.tour[0])
        return self.tour_length


class Colony:
    def __init__(self, tsp, rng, n, alpha, beta, rho):
        self.tsp = tsp
        self.pheromones = np.identity(len(tsp)) * -1 + 1
        self.heuristic = np.zeros((len(tsp), len(tsp)))
        self.probability = np.zeros((len(tsp), len(tsp)))
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.rng = rng

        # precompute heuristic
        for i, j in np.ndindex(self.heuristic.shape):
            if i != j:
                self.heuristic[i, j] = 1 / self.tsp.dist(i, j)

        self.compute_probability_matrix()
        self.ants = [Ant(tsp, self.probability, rng) for _ in range(n)]

    def compute_probability_matrix(self):
        self.probability = None
        # TODO: Implement here
        # has to be done "inplace" so that it has the same ref and updates for ants

    def evaporate_pheromone(self):
        self.pheromones = None
        # TODO: Implement here
        # has to be done "inplace" so that it has the same ref and updates for ants

    def deposit_pheromone(self):
        self.pheromones = None
        # TODO: Implement here
        # has to be done "inplace" so that it has the same ref and updates for ants


def main(args, rng):
    # Init TSP instance
    tsp = parseTSP.TSP(args.file)

    # For visualization:
    # viz = animation.Visu(tsp)

    # create ant colony
    colony = Colony(tsp, rng, args.ants, args.alpha, args.beta, args.rho)

    # init some parameters
    it = 0
    tours = 0
    best_tour_length = np.inf
    best_tour = []
    ended = False

    # begin swarming
    while not ended:
        for ant in colony.ants:
            ant.search()
            ant_tour = ant.compute_tour_length()
            if ant_tour < best_tour_length:
                best_tour_length = ant_tour
                best_tour = ant.tour
            tours += 1
        print(f"** Tours: {tours} - Best: {best_tour_length} - {best_tour}")
        it += 1

        colony.evaporate_pheromone()
        colony.deposit_pheromone()
        colony.compute_probability_matrix()

        if tours > args.tours or it > args.it:
            ended = True

        # For visualization:
        # viz.animate(best_tour, colony.pheromones)

    print(f"Best: {best_tour_length}, End of ACO")
    input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process args for ACO run")
    parser.add_argument("--ants", help="Nb of ants in colony", type=int, default=10)
    parser.add_argument("--alpha", help="Alpha parameter", type=float, default=1.0)
    parser.add_argument("--beta", help="Beta parameter", type=float, default=1.0)
    parser.add_argument("--rho", help="Rho parameter", type=float, default=0.2)
    parser.add_argument(
        "--tours", help="Maximum number of tours to build", type=int, default=10000
    )
    parser.add_argument(
        "--it", help="Maximum number of iterations to perform", type=int, default=100000
    )
    parser.add_argument("--seed", help="Random seed generator", type=int)
    parser.add_argument("--file", help="Path to the instance file", required=True)

    args = parser.parse_args()

    print("parameters:")
    print(args)

    if args.seed:
        rng = np.random.default_rng(args.seed)
    else:
        rng = np.random.default_rng()
        # random.seed(args.seed)

    main(args, rng)
