#!/usr/bin/python3

# Created on: May 2021
# Author: Ken Hasselmann <ken.hasselmann * ulb.be>

import math
import sys
import numpy as np


class Point2D:
    def __init__(self, id, x, y):
        self.id = id - 1  # to be 0 indexed
        self.x = x
        self.y = y

    def dist_to(self, point):
        return math.sqrt((point.x - self.x) ** 2 + (point.y - self.y) ** 2)

    def __str__(self):
        return f"ID: {self.id} - x={self.x}, y={self.y}"


class TSP:
    def __init__(self, file):
        self.nodes = []
        self.parse_file(file)

        # precompute distance makes code much faster
        self.dist_mat = np.zeros((len(self.nodes), len(self.nodes)))
        for a in range(len(self.nodes)):
            for b in range(len(self.nodes)):
                self.dist_mat[a, b] = self.nodes[a].dist_to(self.nodes[b])

    def __len__(self):
        return len(self.nodes)

    def dist(self, a, b):
        return self.dist_mat[a, b]

    def parse_file(self, file):
        coord_sec = False
        with open(file) as f:
            for line in f:
                if coord_sec == True:
                    if "EOF" in line:
                        return
                    self.nodes.append(Point2D(*map(int, line.split(" "))))
                elif "NAME" in line:
                    self.name = line.split(":")[1].strip()
                elif "TYPE" in line:
                    continue
                elif "COMMENT" in line:
                    continue
                elif "DIMENSION" in line:
                    self.n = int(line.split(":")[1])
                elif "EDGE_WEIGHT_TYPE" in line:
                    e_type = line.split(":")[1].strip()
                    if e_type != "EUC_2D":
                        print(f"Edge type not supported {e_type}")
                elif "NODE_COORD_SECTION" in line:
                    coord_sec = True

    def __str__(self):
        t = f"name: {self.name}, n: {self.n}\n"
        for n in self.nodes:
            t += str(n) + "\n"
        return t


if __name__ == "__main__":
    testInst = TSP(sys.argv[1])
    print(testInst)
