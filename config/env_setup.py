import gym


def load_gym_env(env_name):
    env = gym.make(env_name)
    N_State = env.observation_space.shape[0]
    N_Action = env.action_space.n
    return env,N_State,N_Action

def reset_env(env):
    s = env.reset()
    return s


def take_action(env,state,action):
    s_, r, done, info = env.step(action)
    return s_, r, done, info


