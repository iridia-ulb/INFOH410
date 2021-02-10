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
    # print(queens)
    # print_board(queens, n)

    # we use as heuristic, the number of pair of queens attacking each other
    best_score = h(queens)

    while best_score != 0:
        bests = []
        sideways = []
        current_best = best_score
        for ne in neighbors(queens, n):
            score = h(ne)
            if score < current_best:
                current_best = score
                bests.clear()
                bests.append(ne)
            elif score == best_score:
                sideways.append(ne)  # we save sideways moves
            elif score == current_best:
                bests.append(ne)
        # print("b:", bests, current_best)
        # print("side:", sideways, best_score)
        if bests != []:
            # queens = bests[0]
            queens = random.choice(bests)
            best_score = current_best
        else:
            queens = random.choice(sideways)

    print(best_score, queens)
    print_board(queens, n)


def neighbors(queens, n):
    """
    we define the neighborhood of a solution: one possible neighborhood
    is moving one queen somewhere else of the board
    """
    neighbors = []
    for i in range(len(queens)):
        for x in range(0, n):
            l = queens[:i] + (x,) + queens[i + 1 :]
            neighbors.append(tuple(l))
    return neighbors


def q_exhaustive(n):
    print("Exhaustive search:")
    queens = tuple([i for i in range(n)])
    best = h(queens)
    sol = queens
    for ne in itertools.permutations(queens):
        score = h(ne)
        if score < best:
            best = score
            sol = ne
        if score == 0:
            break
    print_board(sol, n)
    print(best, sol)


def q_bfs(n):
    queens = tuple([random.randint(0, n - 1) for _i in range(n)])
    best_score = h(queens)
    candidate = [queens]
    sol = None
    while candidate != []:
        c = candidate.pop()
        score = h(c)
        if score == 0:
            best_score = score
            sol = c
            break
        for ne in neighbors(c, n):
            candidate.insert(0, ne)

    print(best_score, sol)
    print_board(sol, n)


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
        move = minimax(state, 0)
        # print(move)
        state = move[1]


def minimax(state, player):
    val = has_winner(state)
    # print(state, val)
    if val == -1 or val == 0 or val == 1:
        return [val, None]

    moves = []
    for move in next_moves(state, player):
        m = minimax(move, (player + 1) % 2)
        m[1] = move
        moves.append(m)
    if player == 0:
        move = max(moves)
    else:  # if player == 1:
        move = min(moves)
    return move


def next_moves(state, player):
    moves = []
    for i, l in enumerate(state):  # continue
        for j, s in enumerate(l):
            if s == ".":
                s2 = [s[:] for s in state[:]]
                s2[i][j] = "X" if player == 0 else "O"
                moves.append(s2)
    return moves


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


def is_valid_move(state, x, y):
    if x < 0 or x > 2 or y < 0 or y > 2:
        return False
    elif state[x][y] != ".":
        return False
    else:
        return True


def print_ttt(state):
    for l in state:
        print("| ", end="")
        for v in l:
            print(v, end=" | ")
        print()
        print(" __  __  __")


if __name__ == "__main__":
    # q(20)
    # q_bfs(4)
    # q_exhaustive(20)

    q_minimax()
