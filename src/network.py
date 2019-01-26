import tensorflow as tf
import pandas as pd
import numpy as np
from config.config import GLOBAL_NET_SCOPE
from config.env_setup import env,N_Action,N_State


class ACNetwork(object):

    def __init__(self,scope,global = None):
        # 初始化一个global_network ————只有state输入&基础前馈架构,不需要训练
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32 ,[None,N_State] ,'State_input' )
                action_pro, state_value, public_params, actor_params, critic_params= self._build_net(scope)
        else:
            with tf.variable_scope(scope):
                #  local_network需要训练 所以定义如下几个输入
                self.s = tf.placeholder(tf.float32 ,[None,N_State] ,'State_input')
                self.action_done = tf.placeholder(tf.int32 ,[None,N_Action],'Action_input')
                self.V_target = tf.placeholder(tf.float32,[None,1],'Value_target')
                self.action_pro,self.V_estimated,self.public_params,self.actor_params,self.critic_params = self._build_net(scope)
                # 定义 critic_network loss
                with tf








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


