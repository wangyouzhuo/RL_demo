from cusiroty_driven_agent.A3C_network import ACNetwork,worker
from config.config import GLOBAL_NET_SCOPE,LR_Actor,LR_Critic,LR_Shared
import tensorflow as tf
import threading
from config.config import plus_global_episode_count,reset_global_episode_count,get_global_episode_count
from config.env_setup import load_gym_env,get_dim


if __name__ =="__main__":

    N_Action,N_State= get_dim('CartPole-v0')

    reset_global_episode_count()

    N_WORKERS = 10

    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_Actor = tf.train.RMSPropOptimizer(LR_Actor,name = "RMSPropActor")
        OPT_Critic = tf.train.RMSPropOptimizer(LR_Critic,name = "RMSPropCritic")
        OPT_Shared = tf.train.RMSPropOptimizer(LR_Shared,name = "RMSPropShared")

        OPT_LIST = [OPT_Actor,OPT_Critic,OPT_Shared]

        GLOBAL_ACNet = ACNetwork(session=SESS,scope=GLOBAL_NET_SCOPE,n_s=N_State,n_a=N_Action)

        COORD = tf.train.Coordinator()

        # 在计算图中定义多个workers
        workers = []
        for i in range(N_WORKERS):
            env = load_gym_env('CartPole-v0')
            worker_name = "No.%s_worker"%i
            if i ==5:
                whe_show = True
            else:
                whe_show = False
            workers.append(worker(name=worker_name,global_ACNet=GLOBAL_ACNet,
                                  sess = SESS,optimizer_list=OPT_LIST
                                  ,coordinator=COORD,env = env,
                                  n_s=N_State,n_a=N_Action,
                                  whe_show=whe_show))

        SESS.run(tf.global_variables_initializer())

        workers_threads = []

        # 定义线程列表
        for worker in workers:
            t = threading.Thread(target=worker.work)
            workers_threads.append(t)

        # 开启线程列表中的每个线程
        for thread in workers_threads:
            print("开始线程:",thread)
            thread.start()

        # 让列表中的线程结束后才继续向下
        COORD.join(workers_threads)

        reset_global_episode_count()

        print("所有线程结束！")


