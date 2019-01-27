import tensorflow as tf
import pandas as pd
import numpy as np
from config.config import GLOBAL_NET_SCOPE,ENTROPY_BETA
from config.env_setup import env,N_Action,N_State


class ACNetwork(object):

    def __init__(self,scope,global_ACNet = None,session = None):
        # 初始化一个global_network ————只有state输入&基础前馈架构,不需要训练
        self.sess = session

        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32 ,[None,N_State] ,'State_input' )
                action_pro, state_value, public_params, actor_params, critic_params= self._build_net(scope)
        else:
            with tf.variable_scope(scope):
                #  local_network需要训练 所以定义如下几个输入
                self.s = tf.placeholder(tf.float32 ,[None,N_State] ,'State_input')
                self.action_done = tf.placeholder(tf.int32 ,[None,N_Action],'Action_input')
                self.v_target = tf.placeholder(tf.float32,[None,1],'Value_target')
                self.action_pro,self.v_estimated,self.public_params,self.actor_params,self.critic_params = self._build_net(scope)
                # 定义 critic_network loss
                with tf.name_scope('critic_loss'):
                    td_error = tf.subtract(self.v_target,self.v_estimated ,name = "td_error")
                    self.critic_loss = tf.reduce_mean(tf.square(td_error))
                    self.critic_grad = tf.gradients(self.critic_loss ,self.critic_params)
                # 定义actor_network loss =
                with tf.name_scope('actor_loss'):
                    advantage = td_error
                    # axis = 1计算行向量的平均值
                    log_prob = tf.reduce_sum(tf.log(self.action_pro)*tf.one_hot(self.action_pro,N_Action,dtype=tf.float32),axis=1,keepdims=True)
                    # 如果使用 exp_v = log_prob*tf.advantage ，那么在BP的时候：
                    # actor网络中的experience ，会通过BP对critic的网络参数进行更新
                    # 所以在这一步，要把advantage进行截断（从而不对advantage的参数产生影响）
                    experience = log_prob*tf.stop_gradient(advantage)
                    """
                    actor网络的更新是为了增大 p(action)*advantage。我们在这个引入一个trick,把增大的目标改为 p(action)*advantage + entropy
                    背后的含义是：一味增大原有的目标会使得我们越来越专注于某个固定的动作，从而减弱对环境的探索。如果我们在增大目标的同时增大一个“动作分布的熵”
                    就一定程度上更大范围的探索了环境
                    """
                    entropy = -tf.reduce_sum(self.action_pro*tf.log(self.action_pro),axis=1,keepdims=True)
                    self.actor_loss = ENTROPY_BETA*entropy + experience
                    self.actor_grad = tf.gradients(self.actor_loss,self.actor_params)
                with tf.name_scope('shared_loss'):
                    self.public_grads_from_actor = tf.gradients(self.actor_loss , self.public_params)
                    self.public_grads_from_critic = tf.gradients(self.critic_loss , self.public_params)
                #每个local_network还具有 “push经验给global_network” ，“从global_network中pull参数给自己”的功能
                with tf.name_scope('sync'): # 从global_network中pull参数给自己
                    with tf.name_scope('pull'):
                        #  x.assing(y)   把y赋值给x
                        self.pull_actor_params_op = [ local_params.assign(global_params) for local_params,global_params in zip(self.actor_params,global_ACNet.actor_params)]
                        self.pull_critic_params_op = [ local_params.assign(global_params) for local_params,global_params in zip(self.critic_params,global_ACNet.critic_params)]
                        self.pull_shared_params_op = [ local_params.assign(global_params) for local_params,global_params in zip(self.public_params,global_ACNet.public_params)]

                    with tf.name_scope('push'):   # push经验给global_network 即 用local的梯度更新global
                        # OPA_AC 是针对不同AC设计的优化器
                        self.update_global_actor  = OPT_Actor.apply_gradients(zip(self.actor_grad ,global_ACNet.actor_params))
                        self.update_global_critic = OPT_Critic.apply_gradients(zip(self.critic_grad ,global_ACNet.critic_params))
                        self.update_global_shared_from_actor = OPT_Shared.apply_gradients(zip(self.public_grads_from_actor ,global_ACNet.public_params))
                        self.update_global_shared_from_critic = OPT_Shared.apply_gradients(zip(self.public_grads_from_critic ,global_ACNet.public_params))


    def _build_net(self,scope):
        general_initializer = tf.random_normal_initializer(0.0,0.1)
        with tf.variable_scope('Shared_layer'):
            state_encoder = tf.layers.dense(inputs=self.s,units=200,activation=tf.nn.relu,kernel_initializer=general_initializer,name = 'shared_dense')
        with tf.variable_scope('Actor_layer'):
            action_pro = tf.layers.dense(inputs=state_encoder,units=N_Action,activation=tf.nn.softmax,kernel_initializer=general_initializer ,name = 'actor_dense')
        with tf.variable_scope('Critic_layer'):
            state_value =  tf.layers.dense(inputs=state_encoder,units=1,kernel_initializer=general_initializer,name='critic_dense')
        public_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope = scope+'/Shared_layer')
        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope = scope+'/Actor_layer')
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope = scope +'/Critic_layer')
        return action_pro , state_value ,public_params ,actor_params ,critic_params

    # 用local自己的梯度更新global的参数
    def update_global_params(self,feed_dict):
        self.sess.run([self.update_global_actor , self.update_global_critic ,self.update_global_shared_from_actor,self.update_global_shared_from_critic],feed_dict)

    # 获取global的参数 ，用global的参数更新自己的参数
    def pull_params_from_global(self):
        self.sess.run([self.pull_shared_params_op,self.pull_actor_params_op,self.pull_critic_params_op])

    #
    def choose_actin(self,s):
         action_prob = self.sess.run(self.action_pro , feed_dict= {self.s:s[np.newaxis,:]})
         action = np.random.choice(range[N_Action],p = action_prob.ravel())
         return action

class worker(object):

    def __init__(self,name,global_ACNet,sess):
        self.env = None
        self.name = name
        self.sess = sess
        self.AC_Net = ACNetwork(name,global_ACNet = global_ACNet,session = sess)

    def work(self):
        global GLOBAL_EPISODE_COUNT , GLOBAL_REWARD_SUM
        total_step = 1  # 总的步数
        buffer_state,buffer_action,buffer_reward = [],[],[]
        while not COORD.should_stop() and GLOBAL_EPISODE_COUNT <= MAX__EPISODE:
            state = reset_env(self.env)
            rewar_in_episode = 0
            while True:
                 action = self.AC_Net.choose_actin(state)
                 state_next,reward,done,info = take_action(self.env,state)
                 buffer_action.append(action)
                 buffer_reward.append(reward)
                 buffer_state.append(state)
                 if total_step%UPDATE_ITER == 0 or done:
                    # 开始更新参数 ————>  计算 训练需要feed in local_network 的





















