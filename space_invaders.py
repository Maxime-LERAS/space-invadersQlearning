# Based on the excellent
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# and uses Keras.
import argparse
import glob
import os
import pickle
import random
import sys

# import cv2
import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Reshape
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.core import Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam, Adamax, RMSprop

# Script Parameters
input_dim = 80 * 80
gamma = 0.5
update_frequency = 2
learning_rate = 0.002
epsilon = 1
resume = False
render = False

# Initialize
env = gym.make("SpaceInvaders-v0")
number_of_inputs = env.action_space.n
# number_of_inputs = 1
observation = env.reset()
prev_x = None
xs, dlogps, drs, probs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
last_info = -1
nb_kills = 0

train_X = []
train_y = []

yplot = []
y2plot = []
y3plot = []


def space_preprocess_screen(I):
    I = I[34:194, 40:120]
    I = I[::2, ::1, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 255
    # imageio.imwrite('outfile.png', I)
    return I.astype(np.float).ravel()


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# Define the main model (WIP)
def learning_model(input_dim=80 * 80, model_type=1):
    global epsilon
    model = Sequential()
    if model_type == 0:
        model.add(Reshape((1, 80, 80), input_shape=(input_dim,)))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dense(number_of_inputs, activation='softmax'))
        opt = RMSprop(lr=learning_rate)
    else:
        model.add(Reshape((1, 80, 80), input_shape=(input_dim,)))
        model.add(Convolution2D(32, (8, 8), border_mode='same', activation='relu', init='he_uniform'))
        model.add(Convolution2D(64, (4, 4), border_mode='same', activation='relu', init='he_uniform'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', init='he_uniform'))
        model.add(Dense(number_of_inputs, activation='softmax'))
        opt = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    if resume == True:
        model.load_weights('space_model_checkpoint' + str(gamma) + 'h5')
        f = open("epsilon.txt", "r")
        epsilon = float(f.read())
        f.close()
    return model


model = learning_model()

# Begin training
while True:
    if render:
        env.render()
    # Preprocess, consider the frame difference as features
    cur_x = space_preprocess_screen(observation)

    x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
    prev_x = cur_x
    # Predict probabilities from the Keras model
    aprob = ((model.predict(x.reshape([1, x.shape[0]]), batch_size=1).flatten()))
    # print(aprob)
    # aprob = aprob/np.sum(aprob)
    # Sample action
    # action = np.random.choice(number_of_inputs, 1, p=aprob)
    # Append features and labels for the episode-batch
    xs.append(x)
    probs.append((model.predict(x.reshape([1, x.shape[0]]), batch_size=1).flatten()))
    aprob = aprob / np.sum(aprob)
    # print("random" + str(np.random.choice(number_of_inputs)))
    # print("pa random" + str(np.random.choice(number_of_inputs, 1, p=aprob)[0]))
    if random.uniform(0, 1) < epsilon:
        action = np.random.choice(number_of_inputs)
    else:
        action = np.random.choice(number_of_inputs, 1, p=aprob)[0]

    y = np.zeros([number_of_inputs])
    y[action] = 1
    # print action
    dlogps.append(np.array(y).astype('float32') - aprob)
    observation, reward, done, info = env.step(action)
    if last_info == -1:
        last_info = info['ale.lives']
    if last_info > info['ale.lives']:
        last_info = info['ale.lives']
        print("dead" + str(info['ale.lives']))

    reward_sum += reward
    drs.append(reward)
    if done:
        episode_number += 1
        last_info = -1
        nb_kills = 0
        epx = np.vstack(xs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        epdlogp *= discounted_epr
        # Slowly prepare the training batch
        train_X.append(xs)
        train_y.append(epdlogp)
        xs, dlogps, drs = [], [], []
        y3plot.append(epsilon)
        # Periodically update the model
        if episode_number % update_frequency == 0:
            y_train = probs + learning_rate * np.squeeze(np.vstack(train_y))  # Hacky WIP
            # y_train[y_train<0] = 0
            # y_train[y_train>1] = 1
            # y_train = y_train / np.sum(np.abs(y_train), axis=1, keepdims=True)
            print('Training Snapshot:')
            print(y_train)
            model.train_on_batch(np.squeeze(np.vstack(train_X)), y_train)
            if epsilon >= 0.1:
                epsilon -= 0.05
                f = open("epsilon.txt", "w")
                f.write(str(epsilon))
                f.close()
            # Clear the batch
            train_X = []
            train_y = []
            probs = []
            # Save a checkpoint of the model
            os.remove('space_model_checkpoint' + str(gamma) + '.h5') if os.path.exists('space_model_checkpoint' + str(gamma) + '.h5') else None
            model.save_weights('space_model_checkpoint' + str(gamma) + '.h5')
        # Reset the current environment nad print the current results
        running_reward = reward_sum if running_reward is None else running_reward * gamma + reward_sum * (1 - gamma)
        yplot.append(running_reward)
        y2plot.append(reward_sum)


        print('Environment reset imminent. Total Episode Reward: %f. Running Mean: %f with gamma = %f' % (reward_sum, running_reward,gamma))
        reward_sum = 0
        observation = env.reset()
        prev_x = None
    if reward != 0:
        print(('Jour %d du confinement : ' % episode_number) +
              ('Defeat!' if reward == -5 else 'VICTORY!'))
    if episode_number == 1000:
        # https://cmdlinetips.com/2019/10/how-to-make-a-plot-with-two-different-y-axis-in-python-with-matplotlib/
        xplot = np.arange(0, len(yplot), 1)
        # create figure and axis objects with subplots()
        fig, ax = plt.subplots()
        # make a plot
        ax.plot(xplot, yplot, color="red")
        # set x-axis label
        ax.set_xlabel("episodes", fontsize=14)
        # set y-axis label
        ax.set_ylabel("running_mean", color="red", fontsize=14)
        # twin object for two different y-axis on the sample plot
        ax2 = ax.twinx()
        # make a plot with different y-axis using second axis object
        ax2.plot(xplot, y3plot, color="blue")
        ax2.set_ylabel("epsilon", color="blue", fontsize=14)
        # save the plot as a file
        fig.savefig('running_mean'+str(gamma)+ '.jpg',
                    format='jpeg',
                    dpi=100,
                    bbox_inches='tight')
        plt.close()

        #plt.plot(xplot, yplot, label="running_mean")
        #plt.legend()
        #plt.savefig('mean' + str(gamma) + '.png')
        fig, ax = plt.subplots()
        # make a plot
        ax.plot(xplot, y2plot, color="red")
        # set x-axis label
        ax.set_xlabel("episodes", fontsize=14)
        # set y-axis label
        ax.set_ylabel("reward_sum", color="red", fontsize=14)
        # twin object for two different y-axis on the sample plot
        ax2 = ax.twinx()
        # make a plot with different y-axis using second axis object
        ax2.plot(xplot, y3plot, color="blue")
        ax2.set_ylabel("epsilon", color="blue", fontsize=14)
        # save the plot as a file
        fig.savefig('episode_reward' + str(gamma) + '.jpg',
                    format='jpeg',
                    dpi=100,
                    bbox_inches='tight')
        plt.close()
        # plt.close()
        # plt.plot(xplot, y2plot, label="reward_sum")
        # plt.legend()
        # plt.savefig('episodes'+str(gamma)+'.png')
        sys.exit()
