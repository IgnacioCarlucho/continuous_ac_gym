

import tensorflow as tf
import numpy as np


FIRST_LAYER = 500
SECOND_LAYER = 500


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, learning_rate, num_actor_vars, scope = "value_estimator", device = '/cpu:0'):
        with tf.variable_scope(scope):
            self.sess = sess
            self.s_dim = state_dim
            self.learning_rate = learning_rate
            self.device = device
            # Create the critic network
            self.inputs, self.out = self.create_critic_network(scope)

            self.network_params = tf.trainable_variables()[num_actor_vars:]

            # Network target (y_i)
            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
            
            self.loss = tf.squared_difference(self.predicted_q_value, self.out)
            self.optimizer_c = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            
            

    def create_critic_network(self, scope):
        with tf.device(self.device):
            with tf.variable_scope(scope):
                


                # Placeholders
                inputs = tf.placeholder(tf.float32, shape=[None, self.s_dim])
                # Layer 1 without BN
                
                l1 = tf.contrib.layers.fully_connected(inputs,FIRST_LAYER)
                l2 = tf.contrib.layers.fully_connected(l1,SECOND_LAYER)
                out = tf.contrib.layers.fully_connected(l2,1, activation_fn=None)
                
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
        self.saver.save(self.sess,'./critic_model.ckpt')
        #saver.save(self.sess,'actor_model.ckpt')
        print("Model saved in file:")

    
    def recover_critic(self):
        self.saver.restore(self.sess,'./critic_model.ckpt')
        #saver.restore(self.sess,'critic_model.ckpt')
    