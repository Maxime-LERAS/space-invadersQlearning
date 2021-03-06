# -*- coding: utf-8 -*-
# source : https://github.com/keon/deep-q-learning/blob/master/dqn.py
# Based on the excellent
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
import random
import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Input, Dense, Reshape
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.core import Activation, Dropout, Flatten
from keras.optimizers import Adam

EPISODES = 80


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.94):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=8)
        self.gamma = gamma  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.00003
        self.learning_rate = 0.0005
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Reshape((1, 80, 80), input_shape=(80 * 80,)))
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', activation='relu', init='he_uniform'))
        model.add(Convolution2D(64, (4, 4), border_mode='same', activation='relu', init='he_uniform'))
        # model.add(Convolution2D(64, 3))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape([1, state.shape[0]]))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state.reshape([1, next_state.shape[0]])).flatten()[0]))
            target_f = self.model.predict(state.reshape([1, state.shape[0]]))
            target_f[0][action] = target
            self.model.fit(state.reshape([1, state.shape[0]]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    # Parser source : https://github.com/yilundu/DQN-DDQN-on-Space-Invaders/blob/master/main.py
    parser = argparse.ArgumentParser(description="Train and test different networks on Space Invaders")

    parser.add_argument("-g", "--gamma", type=float, action='store',
                        help="Specify the gamma value, between 0 and 1", required=False)

    args = parser.parse_args()
    print(args)

    gamma = args.gamma


    def space_preprocess_screen(I):
        I = I[34:194, 40:120]
        I = I[::2, ::1, 0]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 255
        # imageio.imwrite('outfile.png', I)
        return I.astype(np.float).ravel()


    env = gym.make('SpaceInvaders-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = None
    if gamma is None:
        agent = DQNAgent(state_size, action_size)
    else:
        agent = DQNAgent(state_size, action_size, gamma)
    # agent.load("./spacemodel.h5")
    done = False
    batch_size = 4

    plotRewardMean = []
    plotEpsilon = []
    plotTotalReward = []
    rewardMean = 0
    shiftingRewardMean = deque(maxlen=15)
    for e in range(EPISODES):
        state = env.reset()
        state = space_preprocess_screen(state)
        total_reward = 0
        imageId = 0
        while not done:
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            if reward >= 200:
                reward = reward - 200  # Suppression du bonus (Les bonus ne sont même pas affichés dans la fenêtre)
            total_reward += reward
            next_state = space_preprocess_screen(next_state)
            if imageId % 4 == 0:
                # Frame skipping : taking 1 frame over 4
                # https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
                agent.memorize(state, action, reward, next_state, done)
            state = next_state
            imageId += 1
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}, gamma: {}"
                      .format(e, EPISODES, total_reward, agent.epsilon, agent.gamma))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if rewardMean == 0:
            rewardMean = total_reward
        rewardMean = 0.95 * rewardMean + 0.05 * total_reward
        plotEpsilon.append(agent.epsilon)
        plotTotalReward.append(total_reward)
        shiftingRewardMean.append(total_reward)
        plotRewardMean.append(rewardMean)

        done = False
        if e % 2 == 0:
            agent.save("./lastresults/spacemodel" + str(agent.gamma) + ".h5")


            # sources : https://cmdlinetips.com/2019/10/how-to-make-a-plot-with-two-different-y-axis-in-python-with-matplotlib/
            xplot = np.arange(0, len(plotTotalReward), 1)
            fig, ax = plt.subplots()
            ax.plot(xplot, plotRewardMean, color="red")
            ax.set_xlabel("episodes", fontsize=14)
            ax.set_ylabel("Score mean", color="red", fontsize=14)
            ax2 = ax.twinx()
            ax2.plot(xplot, plotEpsilon, color="blue")
            ax2.set_ylabel("epsilon", color="blue", fontsize=14)
            fig.savefig('./lastresults/meanScore' + str(agent.gamma) + '.jpg',
                        format='jpeg',
                        dpi=100,
                        bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots()
            ax.plot(xplot, plotTotalReward, color="red")
            ax.set_xlabel("episodes", fontsize=14)
            ax.set_ylabel("Score", color="red", fontsize=14)
            ax2 = ax.twinx()
            ax2.plot(xplot, plotEpsilon, color="blue")
            ax2.set_ylabel("epsilon", color="blue", fontsize=14)
            fig.savefig('./lastresults/episode_score' + str(agent.gamma) + '.jpg',
                        format='jpeg',
                        dpi=100,
                        bbox_inches='tight')
            plt.close()
