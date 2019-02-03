import gym


def load_gym_env(env_name):
    env = gym.make(env_name)
    # N_State = env.observation_space.shape[0]
    # N_Action = env.action_space.n
    return env

def reset_env(env):
    s = env.reset()
    return s

def take_action(env,state,action):
    s_, r, done, info = env.step(action)
    return s_, r, done, info


def get_dim(env_name):
    env = gym.make(env_name)
    N_S = env.observation_space.shape[0]
    N_A = env.action_space.n
    return N_A,N_S



def show_or_not(env,whether):
    if whether:
        env.render()