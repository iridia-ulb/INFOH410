#!/usr/bin/python3
import random
import itertools


def q(n):
    """
    We want to solve the N-queens problem: put n queens on a n*n board,
    with no queen attacking each other.
    """
    print("Hill climbing:")
    # We put the queens on each column on the board
    queens = tuple([random.randint(0, n - 1) for _i in range(n)])
    # print_board(queens, n)


def neighbors(queens, n):
    """
    we define the neighborhood of a solution: one possible neighborhood
    is moving one queen somewhere else of the board
    """


def q_exhaustive(n):
    print("Exhaustive search:")


def h(queens):
    counter = 0
    for i in range(len(queens)):
        for j in range(i + 1, len(queens)):
            if queens[i] == queens[j]:
                counter += 1
            elif abs(queens[i] - queens[j]) == abs(i - j):
                counter += 1
    return counter


def print_board(queens, n):
    board = [["." for i in range(n)] for i in range(n)]
    for i in range(len(queens)):
        board[queens[i]][i] = "Q"

    for l in board:
        for c in l:
            print(c, end=" ")
        print()


def q_minimax():
    """
    You are player 0 at a game of tic tac toe, and you want to play against
    the computer using mix-max algorithm
    """
    state = [[".", ".", "."], [".", ".", "."], [".", ".", "."]]

    play = True
    while play:
        val = has_winner(state)
        if val == -1 or val == 1:
            print_ttt(state)
            print(f"Player {val} has won !")
            break
        elif val == 0:
            print_ttt(state)
            print("It's a tie !")
            break

        print_ttt(state)
        # your turn:
        x = int(input("Please enter line number of your move (0-2): "))
        y = int(input("Please enter column number of your move (0-2): "))
        if not is_valid_move(state, x, y):
            print("Invalid move, try again!")
            continue
        state[x][y] = "O"

        # computer turn:
        # move = minimax(state, 0)

        # state = move[1]


def is_valid_move(state, x, y):
    if x < 0 or x > 2 or y < 0 or y > 2:
        return False
    elif state[x][y] != ".":
        return False
    else:
        return True


def has_winner(state):
    for l in state:  # check lines for winner
        if l == ["X"] * 3:
            return 1
        elif l == ["O"] * 3:
            return -1
    for c in [[l[i] for l in state] for i in range(len(state))]:  # cols
        if c == ["X"] * 3:
            return 1
        elif c == ["O"] * 3:
            return -1
    d = [l[i] for i, l in enumerate(state)]  # first diag
    if d == ["X"] * 3:
        return 1
    elif d == ["O"] * 3:
        return -1
    d = [l[-i - 1] for i, l in enumerate(state)]  # second diag
    if d == ["X"] * 3:
        return 1
    elif d == ["O"] * 3:
        return -1
    for l in state:  # continue
        for s in l:
            if s == ".":
                return None
    return 0  # its a tie


def print_ttt(state):
    for l in state:
        print("| ", end="")
        for v in l:
            print(v, end=" | ")
        print()
        print(" __  __  __")


if __name__ == "__main__":
    q(4)
    q_exhaustive(4)

    q_minimax()
