from simple_implement_of_ppo.model import *
from simple_implement_of_ppo.train import *
import gym

if __name__ == "__main__":
    env = gym.make('Pendulum-v0').unwrapped

    N_S, N_A = 3, 1

    ppo = init_ppo_network(name='Fuck_Reinforcement_Learning',
                           KL_Min=0.006666,KL_Max=0.015,KL_BETA=0.5,
                           s_dim=N_S,a_dim=N_A,
                           action_continous=True,
                           LR_A=0.0001,LR_C=0.00015)

    train(EP_MAX=1000,MAX_STEPS=200,BATCH_SIZE=32,env = env,net=ppo)

