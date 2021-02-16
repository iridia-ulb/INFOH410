#!/usr/bin/python3
import gym
import numpy as np
from numpy.random import default_rng
import tensorflow as tf
from collections import deque
import random
from pathlib import Path

LR = 0.1
DISCOUNT = 0.95


def q():
    env = gym.make("CartPole-v0")
    rng = default_rng()

    numBins = 20
    bins = [
        # np.linspace(-4.8, 4.8, numBins),
        np.linspace(-4, 4, numBins),
        np.linspace(-0.418, 0.418, numBins),
        # np.linspace(-4, 4, numBins),
    ]
    features = slice(1, 3)
    qTable = rng.uniform(
        low=0,
        high=0,
        # size=([numBins] * len(env.observation_space.high) + [env.action_space.n]),
        size=([numBins] * len(bins) + [env.action_space.n]),
    )

    print("b:", bins)
    # print("qt: ", qTable[0][0])
    print([numBins] * len(bins) + [env.action_space.n])
    cnt_l = []
    eps = 1

    for i_episode in range(10000):
        observation = env.reset()
        print(observation[features])
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

        if sum(cnt_l) > 100 * 185:
            print("Learned, lets test")
            state = env.reset()
            while True:
                env.render()  # if running RL comment this out
                discreteState = get_discrete_state(state[features], bins, len(bins))
                action = np.argmax(qTable[discreteState])
                state, reward, done, info = env.step(action)

    env.close()


# Given a state of the enviroment, return its descreteState index in qTable
def get_discrete_state(state, bins, obsSpaceSize):
    stateIndex = []
    for i in range(obsSpaceSize):
        stateIndex.append(
            np.digitize(state[i], bins[i]) - 1
        )  # -1 will turn bin into index
    return tuple(stateIndex)


def qNN():
    env = gym.make("CartPole-v0")
    rng = default_rng()
    cnt_l = []

    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(4,)))
    model.add(tf.keras.layers.Dense(24, activation="relu"))
    model.add(tf.keras.layers.Dense(24, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="linear"))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001))
    model.summary()
    print(model.output_shape)

    model_target = tf.keras.models.clone_model(model)

    memory = deque(maxlen=3000)
    eps = 1

    for i_episode in range(10000):
        # print("nepi: ", i_episode)
        state = env.reset()
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        cnt = 0  # how may movements cart has made

        while True:
            # env.render()  # if running RL comment this out
            cnt += 1
            eps -= 0.9 / 5000
            eps = max(eps, 0.1)
            # Get action from Q NN
            # print("st: ", state_tensor)
            action_probs = model(state, training=False)

            # print("e:", eps, " cnt: ", cnt, " ga: ", DISCOUNT)
            if rng.random() > eps:
                action = tf.argmax(action_probs[0]).numpy()
            else:
                action = rng.integers(0, env.action_space.n)

            # perform action on enviroment
            newState, reward, done, info = env.step(action)
            newState = np.reshape(newState, [1, 4])
            # newState = tf.expand_dims(tf.convert_to_tensor(newState), 0)
            memory.append((state, action, reward, newState, done))

            histo = []
            # if cnt % 5 == 0 and len(memory) > 32:
            if len(memory) > 32:
                # batch = memory[-16:]
                batch = random.sample(memory, 32)
                states = []
                targets = []
                for stat, actio, rewar, next_stat, don in batch:
                    # print(state, action, reward, next_state, done)
                    if don:
                        target = -1
                        # print("shoulnd happen!", stat, actio, rewar, don)
                    else:
                        target = rewar + DISCOUNT * np.amax(
                            model_target.predict(next_stat)[0]
                        )
                    # target = R(s,a) + gamma * max Q`(s`,a`)
                    # target (max Q` value) is output of Neural Network which takes s` as an input
                    # amax(): flatten the lists (make them 1 list) and take max value

                    train_target = model.predict(stat)
                    # s --> NN --> Q(s,a)=train_target
                    train_target[0][actio] = target
                    states.append(stat[0])
                    targets.append(train_target[0])

                hist = model.fit(
                    np.array(states), np.array(targets), epochs=1, verbose=0
                )
                histo.append(hist.history["loss"])
                # verbose: dont show loss and epoch
            state = newState

            # print("rew: ", reward, done)

            if done and i_episode % 10 == 0:
                model.save(f"saved/cartpoletest-{i_episode}", save_format="tf")

            if done and i_episode % 5 == 0:
                model_target.set_weights(model.get_weights())

            if done:
                print(
                    f"{i_episode} - done: fell after: {cnt}, {eps} - loss: {np.mean(histo)}"
                )
                cnt_l.append(cnt)
                break

        # # print(cnt_l)
        # if len(cnt_l) > 100:
        #     cnt_l.pop(0)

        # if sum(cnt_l) > 100 * 185:
        #     print("Learned, lets test")
        #     state = env.reset()
        #     while True:
        #         env.render()  # if running RL comment this out
        #         discreteState = get_discrete_state(state[features], bins, len(bins))
        #         action = np.argmax(qTable[discreteState])
        #         state, reward, done, info = env.step(action)

    env.close()


def qNNtest(filename):
    model = tf.keras.models.load_model(filename)
    model.summary()
    print("Learned, lets test")

    env = gym.make("CartPole-v0")

    state = env.reset()
    state = np.reshape(state, [1, 4])

    while True:
        env.render()  # if running RL comment this out
        action_probs = model(state, training=False)
        action = tf.argmax(action_probs[0]).numpy()
        state, reward, done, info = env.step(action)
        state = np.reshape(state, [1, 4])


if __name__ == "__main__":
    # q()
    # qNN()
    qNNtest(Path("saved/cartpoletest-150"))
