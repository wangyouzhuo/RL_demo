
GLOBAL_NET_SCOPE = 'Global_Network'

ENTROPY_BETA = 0.05

MAX__EPISODE = 200000

UPDATE_ITER = 5

MAX_STEP_IN_EPISODE = 2000

LR_Actor = 0.001

LR_Critic = 0.001

LR_Shared = 0.002

global GLOBAL_EPISODE_COUNT







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

