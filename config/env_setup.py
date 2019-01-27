import gym

GAME_NAME = 'CartPole-v0'

env = gym.make(GAME_NAME)

N_State= env.observation_space.shape[0]

N_Action = env.action_space.n

def reset_env(env):
    return


def take_action(env,state,action):
    return


