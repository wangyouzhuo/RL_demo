import tensorflow as tf
import numpy as np
"""
simple implement of PPO in one-thread

"""

A_ITER = 10
C_ITER = 10

class PP0(object):
    def __init__(self,name,s_dim,a_dim,action_continous,LR_A,LR_C,adaptive_dict,THRESHOLD):
        self.THRESHOLD = THRESHOLD
        self.adaptive_dict = adaptive_dict
        """
            key:   KL_Min   KL_Max   KL_BETA
        """
        self.sess = tf.Session()
        self.s_dim,self.a_dim,self.action_continous = s_dim , a_dim ,action_continous
        self.s = tf.placeholder(tf.float32,[None,s_dim],'state')
        self.q_v = tf.placeholder(tf.float32,[None,1],'state_action_value')
        self.a = tf.placeholder(tf.float32,[None,a_dim])
        self.adv_in_aloss = tf.placeholder(tf.float32,[None,1],'advantage')
        self.KL_BETA =  tf.placeholder(tf.float32,None,'KL_PUNISH_BETA')
        self.OPT_A = tf.train.AdamOptimizer(LR_A)
        self.OPT_C = tf.train.AdamOptimizer(LR_C)

        self._build_net(name)
        self._prepare_loss_and_train(name)
        self.network_initial()

    def generate_fc_weight(self, shape,whe_train,name='weight_fc'):
        threshold = 1.0 / np.sqrt(shape[0])
        weight_matrix = tf.random_uniform(shape, minval=-threshold, maxval=threshold)
        weight = tf.Variable(weight_matrix, name=name,trainable=whe_train)
        return weight

    def generate_fc_bias(self, shape,whe_train,name='bias_fc'):
        bias_distribution = tf.constant(0.0, shape=shape)
        bias = tf.Variable(bias_distribution, name=name,trainable=whe_train)
        return bias

    def _distribution_net(self,input,input_dim,n_unit,name,trainable):
        # # continous action space
        with tf.variable_scope(name):
            # state ---> n_unit hidden_states
            w_l1 = self.generate_fc_weight(shape=[input_dim,n_unit],name='w_l1',whe_train=trainable)
            b_l1 = self.generate_fc_bias(shape=[n_unit],name='b_l1',whe_train=trainable)
            w_mean = self.generate_fc_weight(shape=[n_unit, self.a_dim], name='w_mean',whe_train=trainable)
            b_mean = self.generate_fc_bias(shape=[self.a_dim], name='b_mean',whe_train=trainable)
            w_var = self.generate_fc_weight(shape=[n_unit, self.a_dim], name='w_var',whe_train=trainable)
            b_var = self.generate_fc_bias(shape=[self.a_dim], name='b_var',whe_train=trainable)

            hidden_l = tf.nn.relu( tf.matmul(input,w_l1)+b_l1)
            mean     =   2*tf.nn.tanh( tf.matmul(hidden_l,w_mean)+b_mean )
            variance = tf.nn.softplus( tf.matmul(hidden_l,w_var)+b_var)
            distribution = tf.distributions.Normal(loc=mean,scale=variance)
        params = [w_l1,b_l1,w_mean,b_mean,w_var,b_var]
        # with tf.variable_scope(name):
        #     l1 = tf.layers.dense(input, n_unit, tf.nn.relu, trainable=trainable)
        #     mu = 2 * tf.layers.dense(l1, self.a_dim, tf.nn.tanh, trainable=trainable)
        #     sigma = tf.layers.dense(l1, self.a_dim, tf.nn.softplus, trainable=trainable)
        #     norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        # params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return distribution, params

    def _build_net(self,name):
        with tf.variable_scope(name):
            with tf.variable_scope('Critic_Net'):
                # state ---> l_a
                self.w_c = self.generate_fc_weight(shape=[self.s_dim,100],whe_train=True,name='w_c')
                self.b_c = self.generate_fc_bias(shape=[100],whe_train=True,name='b_c')
                l_a = tf.nn.relu(tf.matmul(self.s,self.w_c) + self.b_c)
                self.v = tf.layers.dense(l_a,units=1,activation=None)
            with tf.variable_scope('Actor_Net'):
                # state ---> new_distribution
                self.new_pi,self.new_params = self._distribution_net(input = self.s,input_dim=self.s_dim,n_unit=100,name='New_policy',trainable=True)
                # state ---> old_distribution   (old_pi is just used to stored the params,no need to be trained)
                self.old_pi,self.old_params = self._distribution_net(input = self.s,input_dim=self.s_dim,n_unit=100,name='Old_policy',trainable=False)
            with tf.variable_scope('sample_action'):
                self.sample_action = tf.squeeze(self.new_pi.sample(1),axis=0)
            with tf.variable_scope('sync_policy'):
                self.update_policy = [ old.assign(new) for new,old in zip(self.new_params,self.old_params)]  #tf.assign(ref=new,value=old)

    def _prepare_loss_and_train(self,name):
        with tf.variable_scope(name):
            with tf.variable_scope('Actor_loss'):
                # maximum  the "actor objective function":  [new_pi(s,a)/old_pi(s,a)]*[old_pi_based_advantage(s,a)] - BETA*KL_divergence
                self.new_pi_find_error = self.new_pi.prob(self.a)
                self.old_pi_find_error = self.old_pi.prob(self.a)
                self.pi_objective = ( self.new_pi.prob(self.a)/self.old_pi.prob(self.a) )*self.adv_in_aloss
                self.kl = tf.distributions.kl_divergence(self.old_pi,self.new_pi)
                self.kl_objective = self.KL_BETA*self.kl
                self.a_loss = -tf.reduce_mean(self.pi_objective - self.kl_objective)
                # self.kl_mean is used for the computation of adapative of KL_BETA
                self.kl_mean = tf.reduce_mean(self.kl)
            with tf.variable_scope('Actor_train'):
                self.a_train_op = self.OPT_A.minimize(self.a_loss)

            with tf.variable_scope('Critic_loss'):
                self.advantage = self.q_v - self.v
                self.c_loss = tf.reduce_mean(tf.square(self.advantage))
            with tf.variable_scope('Critic_train'):
                self.c_train_op = self.OPT_C.minimize(self.c_loss)

    def choose_action(self,s):
        # n = -2
        # new_params,old_params,w_c = self.sess.run([self.new_params,self.old_params,self.w_c])
        # new,old,w_c = np.sum(new_params[n]),np.sum(old_params[n]),np.sum(w_c)
        # print("new : %s   old:%s  w_c:%s"%(new,old,w_c))
        s = s[np.newaxis, :]
        action = self.sess.run(self.sample_action, {self.s: s})[0]
        return np.clip(action,-self.THRESHOLD , self.THRESHOLD)

    def get_value(self,s):
        if s.ndim < 2:
            s = s[np.newaxis,:]
        return self.sess.run(self.v,{self.s:s})[0,0]

    def network_initial(self):
        self.sess.run(tf.global_variables_initializer())

    def update_network(self,s,a,q_value):
        self.sess.run(self.update_policy)
        advantage = self.sess.run(self.advantage,{self.s:s ,self.q_v:q_value})
        # update Actor_network
        for _ in range(A_ITER):
            _,kl = self.sess.run([self.a_train_op,self.kl_mean],
                                 {self.s:s  ,  self.adv_in_aloss:advantage,
                                  self.a:a  ,  self.KL_BETA:self.adaptive_dict['KL_BETA']})
        # update Critic_network
        for _ in range(C_ITER):
            self.sess.run(self.c_train_op,{self.s:s,self.q_v:q_value})
        # update adaptive params KL_BETA:
        #     if kl < kl_min ---> means the kl_divergance's power is too stronge , we need to dicrease the KL_BETA
        #     if kl > kl_max ---> means the kl_divergance's power is too weak    , we need to increase the KL_BETA
        if kl < self.adaptive_dict['KL_Min']:
            self.adaptive_dict['KL_BETA'] = self.adaptive_dict['KL_BETA']/2.0
        if kl > self.adaptive_dict['KL_Max']:
            self.adaptive_dict['KL_BETA'] = self.adaptive_dict['KL_BETA']*2.0
        self.adaptive_dict['KL_BETA'] = np.clip(self.adaptive_dict['KL_BETA'],1e-4,20)



















