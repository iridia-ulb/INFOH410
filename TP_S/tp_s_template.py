#!/usr/bin/python3
import queue
import heapq
from collections import deque


def q1():
    print("Q1:")
    # using a stack:
    d = []
    # ...

    # using a queue
    # ...

    # using a PrioQueue or heapq
    # ...


def q3():
    """the graph can be stored using a adjency list or an adjency matrix.
    Usually, the matrix is easier to use but uses more memory, here we use
    an adjency list.
    We store states as indexes in the list, where S->0, A->1, B->2 etc.
    each sublist represents the list all states to which state i is
    connected, with associated cost."""
    print("Q2:")

    graph = [
        [(2, 7), (1, 3)],  # S
        [(4, 6), (3, 1)],  # A
        [(7, 9), (5, 1)],  # B
        [(0, 2), (4, 4)],  # C
        [(6, 6), (2, 3)],  # D
        [(7, 5)],  # E
        [(3, 2)],  # G1
        [(2, 8)],  # G2
    ]
    weights = [10, 5, 8, 3, 2, 4, 0, 0]

    # your search:
    # ...


def q4():
    """
    We represent the state by a tuple of 3 values,
    (x,y,z) where x is the number of missionaries at the left
    y is the number of cannibals at the right and z is the position
    of the boat. The initial state is (3,3,0).
    """
    print("Q3:")
    s = (3, 3, 1)
    # ...


def q5():
    """
    We want to find the shortest path between A and B, we are using A*.
    in A*, f = g+h is the function to minimize where g is the value of
    smallest known path and h in the heuristic.
    """
    print("Q5:")
    grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    start = (3, 1)
    goal = (3, 4)

    # Your search:
    # ...


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


if __name__ == "__main__":
    q1()
    q3()
    q4()
    q5()
