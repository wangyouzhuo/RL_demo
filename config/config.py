
GLOBAL_NET_SCOPE = 'Global_Network'

ENTROPY_BETA = 0.005

MAX__EPISODE = 200000

UPDATE_ITER = 10

MAX_STEP_IN_EPISODE = 2000

LR_Actor = 0.001

LR_Critic = 0.001

LR_Shared = 0.002

global GLOBAL_EPISODE_COUNT






"""
对 全局变量GLOBAL_EPISODE_COUNT
进行修改的几个op
"""
def plus_global_episode_count():
    global GLOBAL_EPISODE_COUNT
    GLOBAL_EPISODE_COUNT =+ 1
    return

def reset_global_episode_count():
    global GLOBAL_EPISODE_COUNT
    GLOBAL_EPISODE_COUNT = 0
    return

def get_global_episode_count():
    global GLOBAL_EPISODE_COUNT
    return GLOBAL_EPISODE_COUNT

