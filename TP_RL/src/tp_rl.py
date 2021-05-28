#!/usr/bin/python3
import gym
import numpy as np
from numpy.random import default_rng

LR = 0.1
DISCOUNT = 0.95


def q():
    env = gym.make("CartPole-v1")
    rng = default_rng()

    numBins = 25
    bins = [
        np.linspace(-4.8, 4.8, numBins),
        # np.linspace(-5, 5, numBins),
        np.linspace(-0.418, 0.418, numBins),
        # np.linspace(-5, 5, numBins),
    ]
    features = slice(1, 3)
    # features = slice(0, 4)

    qTable = rng.uniform(
        low=0,
        high=0,
        size=([numBins] * len(bins) + [env.action_space.n]),
    )

    print("b:", bins)
    # print("qt: ", qTable[0][0])
    print([numBins] * len(bins) + [env.action_space.n])
    cnt_l = []
    eps = 1

    for i_episode in range(10000):
        observation = env.reset()
        # print(observation[features])
        discreteState = get_discrete_state(observation[features], bins, len(bins))
        cnt = 0  # how may movements cart has made

        while True:
            # env.render()  # if running RL comment this out
            cnt += 1
            eps -= 0.02
            eps = max(eps, 0.1)

            if rng.random() > eps:
                # Get action from Q table
                action = np.argmax(qTable[discreteState])
            else:
                # Get random action
                action = rng.integers(0, env.action_space.n)

            # perform action on enviroment
            newState, reward, done, info = env.step(action)

            newDiscreteState = get_discrete_state(newState[features], bins, len(bins))

            maxFutureQ = np.max(
                qTable[newDiscreteState]
            )  # estimate of optiomal future value
            currentQ = qTable[discreteState + (action,)]  # old value

            # formula to caculate all Q values
            newQ = (1 - LR) * currentQ + LR * (reward + DISCOUNT * maxFutureQ)
            # or:
            # newQ = reward + DISCOUNT * maxFutureQ

            # Update qTable with new Q value
            qTable[discreteState + (action,)] = newQ
            discreteState = newDiscreteState
            if done:
                print(f"Done: fell after: {cnt}")
                cnt_l.append(cnt)
                break

        # print(cnt_l)
        if len(cnt_l) > 100:
            cnt_l.pop(0)

        if sum(cnt_l) > 100 * 180:
            print("Learned, lets test")
            state = env.reset()
            while True:
                env.render()
                discreteState = get_discrete_state(state[features], bins, len(bins))
                action = np.argmax(qTable[discreteState])
                state, reward, done, info = env.step(action)

    env.close()


# Given a state of the enviroment, return its discreteState index in qTable
def get_discrete_state(state, bins, obsSpaceSize):
    stateIndex = []
    for i in range(obsSpaceSize):
        stateIndex.append(
            np.digitize(state[i], bins[i]) - 1
        )  # -1 will turn bin into index
    return tuple(stateIndex)


if __name__ == "__main__":
    q()
