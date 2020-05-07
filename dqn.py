# -*- coding: utf-8 -*-
# source : https://github.com/keon/deep-q-learning/blob/master/dqn.py
# Based on the excellent
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Input, Dense, Reshape
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.core import Activation, Dropout, Flatten
from keras.optimizers import Adam

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.0002
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Reshape((1, 80, 80), input_shape=(80*80,)))
        model.add(Convolution2D(32, 9, 9, subsample=(4, 4), border_mode='same', activation='relu', init='he_uniform'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', init='he_uniform'))
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
    # print(env.observation_space)
    # print(state_size)
    # state_size = np.reshape(state_size, [1, state_size])
    # print(space_preprocess_screen(env.observation_space))
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    plotRewardMean = []
    plotEpsilon = []
    plotTotalReward = []

    rewardMean = 0
    for e in range(EPISODES):
        state = env.reset()
        # print(state)
        state = space_preprocess_screen(state)
        # print(state)
        # state = np.reshape(state, [1, state_size])
        total_reward = 0

        while not done:
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            # print(next_state)
            next_state = space_preprocess_screen(next_state)
            # next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, total_reward, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        plotEpsilon.append(agent.epsilon)
        plotTotalReward.append(total_reward)
        plotRewardMean.append(np.mean(plotTotalReward))

        done = False
        if e % 3 == 0:
            agent.save("spacemodel"+str(agent.gamma)+".h5")
            xplot = np.arange(0, len(plotTotalReward), 1)
            # create figure and axis objects with subplots()
            fig, ax = plt.subplots()
            # make a plot
            ax.plot(xplot, plotRewardMean, color="red")
            # set x-axis label
            ax.set_xlabel("episodes", fontsize=14)
            # set y-axis label
            ax.set_ylabel("Score mean", color="red", fontsize=14)
            # twin object for two different y-axis on the sample plot
            ax2 = ax.twinx()
            # make a plot with different y-axis using second axis object
            ax2.plot(xplot, plotEpsilon, color="blue")
            ax2.set_ylabel("epsilon", color="blue", fontsize=14)
            # save the plot as a file
            fig.savefig('meanScore' + str(agent.gamma) + '.jpg',
                        format='jpeg',
                        dpi=100,
                        bbox_inches='tight')
            plt.close()

            # plt.plot(xplot, yplot, label="running_mean")
            # plt.legend()
            # plt.savefig('mean' + str(gamma) + '.png')
            fig, ax = plt.subplots()
            # make a plot
            ax.plot(xplot, plotTotalReward, color="red")
            # set x-axis label
            ax.set_xlabel("episodes", fontsize=14)
            # set y-axis label
            ax.set_ylabel("Score", color="red", fontsize=14)
            # twin object for two different y-axis on the sample plot
            ax2 = ax.twinx()
            # make a plot with different y-axis using second axis object
            ax2.plot(xplot, plotEpsilon, color="blue")
            ax2.set_ylabel("epsilon", color="blue", fontsize=14)
            # save the plot as a file
            fig.savefig('episode_score' + str(agent.gamma) + '.jpg',
                        format='jpeg',
                        dpi=100,
                        bbox_inches='tight')
            plt.close()
