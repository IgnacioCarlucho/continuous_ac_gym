
import tensorflow as tf
import numpy as np
# ===========================
#   Actor and Critic DNNs
# ===========================


FIRST_LAYER = 150
SECOND_LAYER = 150

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, action_bound, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.sess = sess
            self.s_dim = state_dim
            self.a_dim = action_dim
            self.learning_rate = learning_rate
            self.action_bound = action_bound

            # Actor Network
            self.inputs, self.out, self.normal_dist = self.create_actor_network(scope)

            self.network_params = tf.trainable_variables()

            # Or loss and apply gradients
            self.td_error = tf.placeholder(dtype=tf.float32, name="advantage")
            self.action_history = tf.placeholder(dtype=tf.float32, name="action_history")
            
            self.loss = -self.normal_dist.log_prob(self.action_history)*self.td_error - 0.1*self.normal_dist.entropy()

            self.optimizer = tf.train.AdamOptimizer(learning_rate)#tf.train.RMSPropOptimizer(learning_rate) 
            self.train_op = self.optimizer.minimize(self.loss)
            
            self.num_trainable_vars = len(
                self.network_params)             

    def create_actor_network(self,scope):
        with tf.variable_scope(scope):
            # weights initialization
            w1_initial = np.random.normal(size=(self.s_dim,FIRST_LAYER)).astype(np.float32)
            w2_initial = np.random.normal(size=(FIRST_LAYER,SECOND_LAYER)).astype(np.float32)
            #w3_initial = np.random.normal(size=(SECOND_LAYER,self.a_dim)).astype(np.float32)

            #w1_initial = np.random.uniform(size=(self.s_dim,FIRST_LAYER),low= -0.01, high=0.01 ).astype(np.float32)  
            #w2_initial = np.random.uniform(size=(FIRST_LAYER,SECOND_LAYER),low= -0.01, high=0.01 ).astype(np.float32)  
            w3_initial = np.random.uniform(size=(SECOND_LAYER,self.a_dim),low= -0.001, high=0.001 ).astype(np.float32)
            # Placeholders
            inputs = tf.placeholder(tf.float32, shape=[None, self.s_dim])
            # Layer 1 without BN
            w1 = tf.Variable(w1_initial)
            b1 = tf.Variable(tf.zeros([FIRST_LAYER]))
            z1 = tf.matmul(inputs,w1)+b1
            l1 = tf.nn.relu(z1)
            # Layer 2 without BN
            w2 = tf.Variable(w2_initial)
            b2 = tf.Variable(tf.zeros([SECOND_LAYER]))
            z2 = tf.matmul(l1,w2)+b2
            l2 = tf.nn.relu(z2)
            #output layer
            # sigma            
            w3_sigma = tf.Variable(w3_initial)
            b3_sigma = tf.Variable(tf.zeros([self.a_dim]))  
            sigma  = tf.nn.softplus(tf.matmul(l2,w3_sigma)+b3_sigma) + 1e-5 # I added a small number to avoid making sigma = 0 
            # mu
            w3_mu = tf.Variable(w3_initial)
            b3_mu = tf.Variable(tf.zeros([self.a_dim]))  
            mu  = tf.matmul(l2,w3_mu)+ b3_mu
            #mu  = self.action_bound * tf.nn.tanh(tf.matmul(l2,w3_mu)+b3_mu)# activation
            # normal distribution
            
            normal_dist = tf.contrib.distributions.Normal(mu, sigma)
            out = tf.squeeze(normal_dist._sample_n(1))
            scaled_out = tf.clip_by_value(out, -self.action_bound, self.action_bound)


            self.saver = tf.train.Saver()
            return inputs, out, normal_dist


    

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    

    def train(self, inputs, actions, td_error):
        self.sess.run([self.out, self.loss, self.train_op], feed_dict={
            self.inputs: inputs,
            self.td_error: td_error,
            self.action_history: actions            
        })

    
    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    def save_actor(self):
        self.saver.save(self.sess,'actor_model.ckpt')
        #saver.save(self.sess,'actor_model.ckpt')
        print("Model saved in file: actor_model")

    
    def recover_actor(self):
        self.saver.restore(self.sess,'actor_model.ckpt')
        #saver.restore(self.sess,'critic_model.ckpt')
    