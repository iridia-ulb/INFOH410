#!/usr/bin/python3
import random
from graph import *

def q_kl():
    graph = loadGraph("graph.txt")
    
    # Initial "random" partitioning
    # First half is A, second is B
    for i in range(int(len(graph.vertices)/2)):
        graph.vertices[i].part = "A"
    for i in range(int(len(graph.vertices)/2), len(graph.vertices)):
        graph.vertices[i].part = "B"

    cutsize = float("inf")

    while True:
        # Reset the groups, "unlock" the vertices.
        groupA = list()
        groupB = list()

        # Populate groups A and B
        # > TODO

        gains = list() # list of swapping gains
        swaps = list() # [ [vertex A, vertex B], ... ], same order as the gains

        # Compute the initial cutsize of the partition
        # > TODO

        while len(groupA) > 0 and len(groupB) > 0:

            potentialGains = list() # list of gains
            potentialSwaps = list() # [[a,b], ...], same order as potentialGains

            # For each pair of vertices, compute the gain if they were swapped.
            # > TODO

            # Only keep the maximum gain, even if it's negative
            # > TODO

            # Remove the vertices from their groups, they are now "locked".
            #TODO

        print("Initial cutsize: {}".format(cutsize))

        # Update the cutsize
        # > TODO

        print("New cutsize: {}\n".format(cutsize))

        # If the cut is not improved, stop.
        # > TODO

    print("The best I cant do is {}".format(cutsize))

if __name__ == "__main__":

    q_kl()