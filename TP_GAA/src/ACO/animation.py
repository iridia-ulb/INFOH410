#!/usr/bin/python3

# Created on: May 2021
# Author: Ken Hasselmann <ken.hasselmann * ulb.be>

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.path as mpath
import sys


class Visu:
    def __init__(self, tsp):
        self.tsp = tsp

        self.fig, self.ax = plt.subplots()
        x, y = self.get_all_cities()
        self.scatter = self.ax.scatter(x, y)

        self.draw_pheromones()

        self.fig.canvas.draw()

    def get_all_cities(self):
        return zip(*[(i.x, i.y) for i in self.tsp.nodes])

    def loop_func(self, i):
        pass

    def draw_pheromones(self):
        self.phe = []
        for i, city in enumerate(self.tsp.nodes):
            for j, other_city in enumerate(self.tsp.nodes):
                x, y = ([city.x, other_city.x], [city.y, other_city.y])
                self.phe.append(
                    lines.Line2D(x, y, alpha=0.01, color="grey", linewidth=5.0)
                )
                self.ax.add_line(self.phe[-1])

    def animate(self, tour, phe):
        x, y = zip(
            *[(self.tsp.nodes[i].x, self.tsp.nodes[i].y) for i in tour + [tour[0]]]
        )
        line = lines.Line2D(x, y)
        self.ax.add_line(line)

        for i, city in enumerate(self.tsp.nodes):
            for j, other_city in enumerate(self.tsp.nodes):
                self.phe[i + j * len(self.tsp)].set_alpha(phe[i, j])

        self.fig.canvas.draw()
        plt.pause(0.001)

        line.remove()


if __name__ == "__main__":
    import parseTSP

    tsp = parseTSP.TSP(sys.argv[1])
    print(tsp)

    viz = Visu(tsp)
    viz.animate()
