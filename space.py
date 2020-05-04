import gym
import imageio
import numpy as np
from gym.utils.play import play

nbimages =0


def mycallbacks(obs_t, obs_tp1, action, rew, done, info):
    global nbimages
    print("action = ", action, " reward = ", rew, "done = ", done, "info", info)
    # imageio.imwrite('test.png', obs_t[34:194:2, 40:120:1, 1])
    with open('X.txt', 'a') as outfileX:
        nbimages+=1
        np.savetxt(outfileX, delimiter='\n', X=obs_t[34:194:4, 12:148:2, 1], fmt='%d')
    with open('Y.txt', 'a') as outfileY:
        np.savetxt(outfileY, delimiter='\n', X=np.array([str(action)]), fmt='%s')


env = gym.make('SpaceInvaders-v0')
env.reset()
gym.utils.play.play(env, zoom=3, fps=12, callback=mycallbacks)

env.close()
