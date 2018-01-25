

import tensorflow as tf
import numpy as np


FIRST_LAYER = 150
SECOND_LAYER = 150


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, learning_rate, num_actor_vars, scope = "value_estimator"):
        with tf.variable_scope(scope):
            self.sess = sess
            self.s_dim = state_dim
            self.learning_rate = learning_rate

            # Create the critic network
            self.inputs, self.out = self.create_critic_network(scope)

            self.network_params = tf.trainable_variables()[num_actor_vars:]

            # Network target (y_i)
            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
            
            self.loss = tf.squared_difference(self.predicted_q_value, self.out)
            self.optimizer_c = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            
            

    def create_critic_network(self, scope):
        with tf.variable_scope(scope):
            # weights initialization
            w1_initial = np.random.normal(size=(self.s_dim,FIRST_LAYER)).astype(np.float32)
            w2_initial = np.random.normal(size=(FIRST_LAYER,SECOND_LAYER)).astype(np.float32)
            #w3_initial = np.random.normal(size=(SECOND_LAYER,1)).astype(np.float32)

            w3_initial = np.random.uniform(size=(SECOND_LAYER,1),low= -0.001, high=0.001 ).astype(np.float32)
            # Placeholders
            inputs = tf.placeholder(tf.float32, shape=[None, self.s_dim], name = 'inputs')
            
            # Layer 1 contains only the inputs of the state
            w1 = tf.Variable(w1_initial)
            b1 = tf.Variable(tf.zeros([FIRST_LAYER]))
            z1 = tf.matmul(inputs,w1) + b1
            l1 = tf.nn.relu(z1)
            # Layer in this layer, the actions are merged as inputs
            w2_i = tf.Variable(w2_initial)
            b2 = tf.Variable(tf.zeros([SECOND_LAYER]))
            z2 = tf.matmul(l1,w2_i) + b2 
            l2 = tf.nn.relu(z2)
            #output layer
            w3 = tf.Variable(w3_initial)
            b3 = tf.Variable(tf.zeros([1]))
            out  = tf.matmul(l2,w3) + b3 # linear activation
            self.saver = tf.train.Saver()
            
            return inputs, out


      
    def train(self, inputs, predicted_q_value):
        return self.sess.run([self.out, self.loss, self.optimizer_c], feed_dict={
            self.inputs: inputs,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

   
    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

       
    def save_critic(self):
        self.saver.save(self.sess,'critic_model.ckpt')
        #saver.save(self.sess,'actor_model.ckpt')
        print("Model saved in file:")

    
    def recover_critic(self):
        self.saver.restore(self.sess,'critic_model.ckpt')
        #saver.restore(self.sess,'critic_model.ckpt')
    