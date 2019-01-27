from src.network import ACNetwork,worker
from config.config import GLOBAL_NET_SCOPE,LR_Actor,LR_Critic,LR_Shared
import tensorflow as tf
import threading

if __name__ =="__main__":
    N_WORKERS = 10

    SESS = tf.Session()

    with tf.device("/cou:0"):
        OPT_Actor = tf.train.RMSPropOptimizer(LR_Actor,name="RMSPropActor")
        OPT_Critic = tf.train.RMSPropOptimizer(LR_Critic,name="RMSPropCritic")
        OPT_Shared = tf.train.RMSPropOptimizer(LR_Shared,name = "RMSPropShared")

        OPT_LIST = [OPT_Actor,OPT_Critic,OPT_Shared]

        GLOBAL_ACNet = ACNetwork(scope=GLOBAL_NET_SCOPE)

        COORD = tf.train.Coordinator()

        # 在计算图中定义多个workers
        workers = []
        for i in range(N_WORKERS):
            worker_name = "No.%s_worker"%i
            workers.append(worker(name=worker,global_ACNet=GLOBAL_NET_SCOPE,sess = SESS,optimizer_list=OPT_LIST,coordinator=COORD))

        SESS.run(tf.global_variables_initializer())

        workers_threads = []

        # 定义线程列表
        for worker in workers:
            t = threading.Thread(target=worker.work)
            workers_threads.append(t)

        # 开启线程列表中的每个线程
        for thread in workers_threads:
            thread.start()

        # 让列表中的线程结束后才继续向下
        COORD.join(workers_threads)

        print("所有线程结束！")


