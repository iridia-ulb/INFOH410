#!/usr/bin/python3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# You should install matplotlib for this code to work
# e.g. pip install matplotlib
# or https://matplotlib.org/users/installing.html


def main():
    """
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
    pass


def find_G(S, n):
    """
    Find the G boundary, the biggest rectangle not containing any
    negative example.
    """
    pass


if __name__ == "__main__":
    main()
