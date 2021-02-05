#!/usr/bin/python3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


def q4():
    """
    Q4 (c)
    In this function, we initialize the positive and negative values,
    and create the basis for plotting the figures.
    Fore info about the plots visit:
    https://matplotlib.org/index.html
    """
    examples_p = [(6, 5), (5, 3), (4, 4)]
    examples_n = [(1, 3), (2, 6), (9, 4), (5, 1), (5, 8)]
    ax = plt.gca()

    for x, y in examples_n:
        ax.scatter(x, y, marker="_")
    for x, y in examples_p:
        ax.scatter(x, y, marker="+")

    a, b, c, d = find_S(examples_p)
    S_bound = patches.Rectangle((a, c), abs(a - b), abs(c - d), fill=False)
    ax.add_patch(S_bound)

    x, y, z, t = find_G((a, b, c, d), examples_n)
    G_bound = patches.Rectangle((x, z), abs(x - y), abs(z - t), fill=False)
    ax.add_patch(G_bound)

    plt.show()


def find_S(p):
    """
    Find the S boundary S = <a,b,c,d> where a <= x <= b and c <= y <= d
    params:
    p: list of positive training examples
    """
    lx = [e[0] for e in p]
    a, b = min(lx), max(lx)
    ly = [e[1] for e in p]
    c, d = min(ly), max(ly)
    print(a, b, c, d)
    return a, b, c, d


def find_G(S, n):
    """
    Find the G boundary, the biggest rectangle not containing any
    negative example.
    """
    a = -math.inf
    b = math.inf
    c = -math.inf
    d = math.inf
    G = [(a, b, c, d)]
    for x, y in n:
        print(G)
        G_p = []
        for h in G:
            if h[0] <= x:
                G_p.append((x + 1, h[1], h[2], h[3]))
            if h[1] >= x:
                G_p.append((h[0], x - 1, h[2], h[3]))
            if h[2] <= y:
                G_p.append((h[0], h[1], y + 1, h[3]))
            if h[3] >= y:
                G_p.append((h[0], h[1], h[2], y - 1))
        G = []
        print("gp:", G_p)
        for h in G_p:
            if compatible(h, S):
                G.append(h)

    print(G)
    return G[0]


def compatible(h, S):
    if h[0] <= S[0] and h[1] >= S[1] and h[2] <= S[2] and h[3] >= S[3]:
        return True
    return False


if __name__ == "__main__":
    q4()
