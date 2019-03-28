from simple_implement_of_ppo.model import PP0
import numpy as np

GAMMA = 0.9



def init_ppo_network(name,KL_Min,KL_Max,KL_BETA,s_dim,
                     a_dim,action_continous
                     ,LR_A,LR_C):
    adaptive_dict = {'KL_Min'  :  KL_Min,
                     'KL_Max'  :  KL_Max,
                     'KL_BETA' :  KL_BETA}
    ppo = PP0(name = name,s_dim = s_dim,a_dim = a_dim,
              action_continous = action_continous,
              LR_A = LR_A,LR_C = LR_C,
              adaptive_dict = adaptive_dict,
              THRESHOLD = 2)
    return ppo


def train(EP_MAX,MAX_STEPS,BATCH_SIZE,env,net):
    ep_reward_list = []
    for ep in range(EP_MAX):
        s = env.reset()
        buffer_s,buffer_a,buffer_r = [],[],[]
        ep_r = 0
        for t in range(MAX_STEPS):
            a = net.choose_action(s)
            s_next,r,done,_ = env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            # buffer_r.append(r)
            buffer_r.append((r + 8) / 8)
            s = s_next
            ep_r = ep_r + r
            if (t+1)%BATCH_SIZE == 0 or t == MAX_STEPS-1 :
                q_value_list = []
                value_next = net.get_value(s_next)
                for r in buffer_r[::-1]:
                    value_next = r + GAMMA*value_next
                    q_value_list.append(value_next)
                q_value_list.reverse()
                bs, ba, br = np.vstack(buffer_s),np.vstack(buffer_a),np.array(q_value_list)[:, np.newaxis]
                net.update_network(bs,ba,br)
                buffer_s, buffer_a, buffer_r = [], [], []
        if ep == 0:
            ep_reward_list.append(ep_r)
        else:
            ep_reward_list.append(ep_reward_list[-1]*0.9 + ep_r*0.1)
        print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        )




