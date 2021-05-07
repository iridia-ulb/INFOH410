#!/usr/bin/python3
import queue
import heapq
from collections import deque


def q1():
    print("Q1:")
    # using a stack: (regular pyuthon list)
    d = []
    d.append(1)
    d.append(3)
    d.append(2)
    print(d.pop())
    print(d.pop())

    # using a queue
    d = queue.Queue()  # can also use deque
    d.put(1)
    d.put(3)
    d.put(2)
    print(d.get())
    print(d.get())

    # using a PrioQueue or heapq
    d = queue.PriorityQueue()
    d.put(1)
    d.put(3)
    d.put(2)
    print(d.get())
    print(d.get())


def q3():
    """the graph can be stored using a adjency list or an adjency matrix.
    Usually, the matrix is easier to use but uses more memory, here we use
    an adjency list."""
    print("Q2:")
    # we store states as indexes in the list, where S->0, A->1, B->2 etc.
    # each sublist represents the list all states to which state i is connected, with
    # associated cost.
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

    # Breadth(Depth) first search:
    q = deque([(0, 0)])
    while len(q) != 0:
        current = q.popleft()  # Breadth
        # current = q.pop()  # Depth
        if current[0] == 7 or current[0] == 6:
            print(f"Found Goal {current[0]} with cost: {current[1]}")
            break
        for neighbors in graph[current[0]]:
            q.append((neighbors[0], current[1] + neighbors[1]))


def q4():
    """
    We represent the state by a tuple of 3 values,
    (x,y,z) where x is the number of missionaries at the left
    y is the number of cannibals at the left and z is the position
    of the boat. The initial state is (3,3,1)
    """
    print("Q3:")
    s = (3, 3, 1)
    q = deque([(s, [s])])
    visited = []
    while len(q) != 0:
        # current = q.popleft()  # Breadth
        current = q.pop()  # Depth
        # print(current)
        if current[0] == (0, 0, 0):
            print(f"Found solution: {current[1]} ")
            break
        for n in neighbor_states(current[0]):
            if n not in visited:
                q.append((n, current[1] + [n]))
        visited.append(current[0])


def neighbor_states(s):
    # we can move 1 missionary and 1 cannibal
    # or 2 cannibals, 2 missionaries or only 1
    admissible = []
    if s[2] == 1:
        futures = [(-1, -1), (-2, 0), (0, -2), (-1, 0), (0, -1)]
    else:
        futures = [(1, 1), (2, 0), (0, 2), (1, 0), (0, 1)]

    for x, y in futures:
        s2 = (s[0] + x, s[1] + y, (s[2] + 1) % 2)
        if (
            s2[0] >= 0
            and s2[1] >= 0
            and s2[0] <= 3
            and s2[1] <= 3
            and (s2[0] >= s2[1] or s2[0] == 0)
            and (3 - s2[0] >= 3 - s2[1] or 3 - s2[0] == 0)
        ):
            admissible.append(s2)

    # print(f"admissible {admissible}")
    return admissible


def q5():
    """
    We want to find the shortest path between A and B, we are using A*.
    in A*, f = g+h is the function to minimize where g is the value of
    smallest known path and h in the heuristic
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

    print_grid(grid)
    q = [(0, start, [start])]
    heapq.heapify(q)

    g_scores = {start: 0}
    while len(q) != 0:
        current = heapq.heappop(q)
        print(current)
        if current[1] == goal:
            print(f"Found solution: {current[1]} ")
            for id in current[2]:
                grid[id[0]][id[1]] = 2
            print_grid(grid)
            break
        for n in moves(current[1], grid):
            g = g_scores[current[1]] + 1
            f = g + manhattan(n, goal)
            if n not in g_scores or g < g_scores[n]:
                heapq.heappush(q, (f, n, current[2] + [n]))
                g_scores[n] = g


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def print_grid(grid):
    for i in grid:
        for j in i:
            if j == 0:
                print("_", end="")
            elif j == 1:
                print("#", end="")
            else:
                print("o", end="")
        print("")


def moves(c, grid):
    # we can move in all 4 directions (no diagonal)
    moves = []
    for i, j in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
        n = (c[0] + i, c[1] + j)
        if (
            n[0] >= 0
            and n[0] < len(grid)
            and n[1] >= 0
            and n[1] < len(grid[0])
            and grid[n[0]][n[1]] == 0
        ):
            moves.append(n)
    return moves


if __name__ == "__main__":
    q1()
    q3()
    q4()
    q5()
