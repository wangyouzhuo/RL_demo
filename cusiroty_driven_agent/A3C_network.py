import tensorflow as tf
import pandas as pd
import numpy as np
from config.config import GLOBAL_NET_SCOPE,ENTROPY_BETA,MAX__EPISODE,MAX_STEP_IN_EPISODE,UPDATE_ITER
from config.config import plus_global_episode_count,reset_global_episode_count,get_global_episode_count
from config.env_setup import take_action,reset_env,load_gym_env,show_or_not


class ACNetwork(object):

    def __init__(self,scope,global_ACNet = None,session = None,
                 optimizer_list = None,n_s = None ,n_a =None):
        # 初始化一个global_network ————只有state输入&基础前馈架构,不需要训练
        self.sess = session
        self.n_s = n_s
        self.n_a = n_a

        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.action = tf.placeholder(tf.int32 ,[None,],'Action_input')
                self.state = tf.placeholder(tf.float32 ,[None,self.n_s] ,'State_input' )
                _,_, self.global_public_params, self.global_actor_params,\
                self.global_critic_params,self.global_predictor_params,_= self._build_net(scope)
        else:
            if optimizer_list : # 针对 actor critic 以及共享层 的 优化器
                self.OPT_Actor  = optimizer_list[0]
                self.OPT_Critic = optimizer_list[1]
                self.OPT_Shared = optimizer_list[2]
            else:
                print("优化器未定义！！")
            with tf.variable_scope(scope):
                #  local_network需要训练 所以定义如下几个输入
                self.state = tf.placeholder(tf.float32 ,[None,self.n_s] ,'State_input')
                self.action = tf.placeholder(tf.int32 ,[None,],'Action_input')
                self.v_target = tf.placeholder(tf.float32,[None,1],'Value_target')
                self.state_next = tf.placeholder(tf.float32,[None,self.n_s],'State_next')

                self.action_pro,self.v_estimated,self.local_public_params,\
                self.local_actor_params,self.local_critic_params ,\
                self.local_predictor_params,self.state_predicted = self._build_net(scope)

                # 定义 critic_network loss
                with tf.name_scope('critic_loss'):
                    self.td_error = tf.subtract(self.v_target,self.v_estimated ,name = "td_error")
                    self.critic_loss = tf.reduce_mean(tf.square(self.td_error))
                    self.critic_grad = tf.gradients(self.critic_loss ,self.local_critic_params)
                # 定义actor_network loss =
                with tf.name_scope('actor_loss'):
                    self.advantage = self.td_error
                    # axis = 1计算行向量的平均值
                    prob = tf.reduce_sum( self.action_pro * tf.one_hot(self.action,self.n_a,dtype=tf.float32) , axis=1 , keepdims=True)
                    # 如果使用 exp_v = log_prob*tf.advantage ，那么在BP的时候：
                    # actor网络中的experience ，会通过BP对critic的网络参数进行更新
                    # 所以在这一步，要把advantage进行截断（从而不对advantage的参数产生影响）
                    experience = prob*tf.stop_gradient(self.advantage)
                    """
                    actor网络的更新是为了增大 p(action)*advantage。我们在这个引入一个trick,把增大的目标改为 p(action)*advantage + entropy
                    背后的含义是：一味增大原有的目标会使得我们越来越专注于某个固定的动作，从而减弱对环境的探索。如果我们在增大目标的同时增大一个“动作分布的熵”
                    就一定程度上更大范围的探索了环境
                    """
                    entropy = -tf.reduce_sum(self.action_pro*tf.log(self.action_pro),axis=1,keepdims=True)
                    self.actor_loss = ENTROPY_BETA*entropy - tf.reduce_mean(experience)
                    #self.actor_loss  = -tf.reduce_mean(experience)
                    self.actor_grad = tf.gradients(self.actor_loss,self.local_actor_params)
                with tf.name_scope('shared_loss'):
                    self.public_grads_from_actor = tf.gradients(self.actor_loss , self.local_public_params)
                    self.public_grads_from_critic = tf.gradients(self.critic_loss , self.local_public_params)
                with tf.name_scope('predictor_loss'):
                    self.predictor_loss = tf.losses.mean_squared_error(self.state_predicted,self.state_next)
                    self.predictor_grad = tf.gradients(self.predictor_loss,self.local_predictor_params)
                #每个local_network还具有 “push经验给global_network” ，“从global_network中pull参数给自己”的功能
                with tf.name_scope('sync'): # 从global_network中pull参数给自己
                    with tf.name_scope('pull'):
                        #  x.assing(y)   把y赋值给x
                        self.pull_actor_params_op = [ local_params.assign(global_params) for local_params,global_params in zip(self.local_actor_params,global_ACNet.global_actor_params)]
                        self.pull_critic_params_op = [ local_params.assign(global_params) for local_params,global_params in zip(self.local_critic_params,global_ACNet.global_critic_params)]
                        self.pull_shared_params_op = [ local_params.assign(global_params) for local_params,global_params in zip(self.local_public_params,global_ACNet.global_public_params)]
                        self.pull_predictor_params_op = [ local_params.assign(global_params) for local_params,global_params in zip(self.local_predictor_params,global_ACNet.global_predictor_params)]
                    with tf.name_scope('push'):   # push经验给global_network 即 用local的梯度更新global
                        # OPA_AC 是针对不同AC设计的优化器
                        self.update_global_actor  = self.OPT_Actor.apply_gradients(zip(self.actor_grad ,global_ACNet.global_actor_params))
                        self.update_global_critic = self.OPT_Critic.apply_gradients(zip(self.critic_grad ,global_ACNet.global_critic_params))
                        self.update_global_shared_from_actor = self.OPT_Shared.apply_gradients(zip(self.public_grads_from_actor ,global_ACNet.global_public_params))
                        self.update_global_shared_from_critic = self.OPT_Shared.apply_gradients(zip(self.public_grads_from_critic ,global_ACNet.global_public_params))
                        self.update_global_predictor = self.OPT_Shared.apply_gradients(zip(self.predictor_grad ,global_ACNet.global_predictor_params))
    # 返回 动作分布，value，共享层参数，actor参数，critic参数
    def _build_net(self,scope):
        general_initializer = tf.random_normal_initializer(0.0,0.1)
        with tf.variable_scope('Shared_layer'):
            state_encoder = tf.layers.dense(inputs=self.state,units=200,activation=tf.nn.relu,kernel_initializer=general_initializer,name = 'shared_dense')
        with tf.variable_scope('Actor_layer'):
            action_pro = tf.layers.dense(inputs=state_encoder,units=self.n_a,activation=tf.nn.softmax,kernel_initializer=general_initializer ,name = 'actor_dense')
        with tf.variable_scope('Critic_layer'):
            state_value = tf.layers.dense(inputs=state_encoder,units=1,kernel_initializer=general_initializer,name='critic_dense')
        with tf.variable_scope('State_predictor'):
            joint_state = tf.concat([self.state,tf.one_hot(self.action,self.n_a,dtype=tf.float32)],1)
            dense_layer = tf.layers.dense(inputs=joint_state ,units=300,activation=tf.nn.relu,kernel_initializer=general_initializer)
            state_predicted = tf.layers.dense(inputs=dense_layer ,units=self.n_s,kernel_initializer=general_initializer)
        public_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope = scope+'/Shared_layer')
        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope = scope+'/Actor_layer')
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope = scope +'/Critic_layer')
        predictor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope = scope +'/State_predictor')
        return action_pro , state_value ,public_params ,actor_params ,critic_params ,predictor_params,state_predicted


    # 用local自己的梯度更新global的参数
    def update_global_params(self,feed_dict):
        self.sess.run([self.update_global_actor , self.update_global_critic ,
                       self.update_global_shared_from_actor,
                       self.update_global_shared_from_critic,
                       self.update_global_predictor],feed_dict)

    # 获取global的参数 ，用global的参数更新自己的参数
    def pull_params_from_global(self):
        self.sess.run([self.pull_shared_params_op,
                       self.pull_actor_params_op,
                       self.pull_critic_params_op,
                       self.pull_predictor_params_op])

    def make_prediction(self,feed_dict):
        state_predicted = self.sess.run([self.state_predicted],feed_dict)
        return state_predicted

    #
    def choose_actin(self,s):
        action_prob = self.sess.run(self.action_pro , feed_dict= {self.state: s[np.newaxis, :]}  )

        # print(s)
        action = np.random.choice(range(action_prob.shape[1]),p=action_prob.ravel())
        return action

class worker(object):

    def __init__(self,name,global_ACNet,sess,optimizer_list,
                 coordinator,env = None,n_a = None,n_s = None,
                 whe_show = None):
        self.threading_coordinator = coordinator
        self.env = env
        self.name = name
        self.sess = sess
        self.AC_Net = ACNetwork(name,global_ACNet = global_ACNet,session = sess,optimizer_list = optimizer_list,n_a=n_a,n_s=n_s)
        show_or_not(env,whe_show)

    def compute_v_target(self,buffer_v_target,buffer_reward,v_next):  # buffer_v_target,buffer_reward,v_next
        gamma = 0.99
        for r in buffer_reward[::-1]:  # reverse buffer r
            v_next = r + gamma * v_next
            buffer_v_target.append(v_next)  # buffer_v_target 和 buffer_r 长度是一样的
        buffer_v_target.reverse()

        # print("---------")
        # print("buffer reward",buffer_reward)
        # print("v_next",v_next)
        # print('v_target',buffer_v_target)
        return buffer_v_target

    def work(self):
        global  GLOBAL_REWARD_SUM
        total_step = 0  # 总的步数
        buffer_state,buffer_action,buffer_reward,buffer_state_next = [],[],[],[]
        while not self.threading_coordinator.should_stop() and get_global_episode_count() <= MAX__EPISODE:
            state = reset_env(self.env)
            reward_in_episode = 0
            curisoty_reward_in_episode = 0
            step_in_episode = 0
            while True:
                action = self.AC_Net.choose_actin(state)
                state_next,reward,done,info = take_action(self.env,state,action)
                feed_dict_of_predictor = {
                    self.AC_Net.action: np.array([action]),
                    self.AC_Net.state: np.array([state]),
                }
                try_prediction = self.AC_Net.make_prediction(feed_dict_of_predictor)
                #print("prediction------",try_prediction[0][0],'\n',"ground_truth----",np.array([state])[0],'\n',"******************************")
                curisoty_reward = 0
                for i in range(self.AC_Net.n_s):
                    curisoty_reward = curisoty_reward + pow((try_prediction[0][0][i]*1.0 - np.array([state])[0][i]),2)
                reward = curisoty_reward*1.0 + reward*1.0
                curisoty_reward_in_episode = curisoty_reward + curisoty_reward_in_episode
                #print(reward)
                #print(curisoty_reward)
                step_in_episode = step_in_episode + 1
                reward_in_episode = reward_in_episode + reward
                buffer_action.append(action)
                buffer_reward.append(reward)
                buffer_state.append(state)
                buffer_state_next.append(state_next)
                if total_step%UPDATE_ITER == 0 or done:
                    # 开始更新参数 ————>  计算 训练需要feed in local_network 的 self.s   self.action   self.v_target
                    #  rt+1 rt+2 rt+3 rt+4 rt+5 如何计算 V_Target t+1 ~ V_Target t+5
                    if done:
                        v_next = 0
                    else:
                        v_next = self.sess.run(self.AC_Net.v_estimated,{self.AC_Net.state:state_next[np.newaxis,:]})[0,0]
                    buffer_v_target = []
                    # print(buffer_reward)
                    buffer_v_target = self.compute_v_target(buffer_v_target,buffer_reward,v_next)
                    buffer_state,buffer_action,buffer_v_target,buffer_state_next = np.array(buffer_state),np.array(buffer_action),np.vstack(buffer_v_target),np.array(buffer_state_next)
                    feed_dict = {
                        self.AC_Net.state : buffer_state,
                        self.AC_Net.action : buffer_action,
                        self.AC_Net.v_target : buffer_v_target,
                        self.AC_Net.state_next : buffer_state_next
                    }
                    feed_dict_of_predictor = {
                        self.AC_Net.action: buffer_action,
                        self.AC_Net.state: buffer_state,
                    }
                    #state_predicted = self.AC_Net.make_prediction(feed_dict_of_predictor)
                    # print("-----------------------")
                    # print("state_predicted",state_predicted)
                    # print("buffer_state_next",buffer_state_next)
                    # print("------------------------")
                    # feed in 数据后 开始更新参数 。在ACNet中，我们更新的是global的参数
                    self.AC_Net.update_global_params(feed_dict)
                    #  global参数得到更新后，把新的global_net参数分发给local_net
                    self.AC_Net.pull_params_from_global()
                    buffer_state, buffer_action, buffer_reward,buffer_state_next = [], [], [],[]
                    # print(buffer_reward)
                state = state_next
                plus_global_episode_count()
                total_step = total_step + 1
                #print(total_step)
                if done or step_in_episode>=MAX_STEP_IN_EPISODE:
                    print(
                        self.name,
                        " ，Episode—", get_global_episode_count(),
                        "| 环境Reward: %s" % step_in_episode,
                        "| 好奇心Reward：%s"% curisoty_reward_in_episode,
                        "| 总Reward：%s"%(reward_in_episode)
                    )
                    plus_global_episode_count()
                    step_in_episode = 0
                    curisoty_reward_in_episode = 0
                    break



































