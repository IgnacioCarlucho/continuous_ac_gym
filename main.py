import numpy as np
import tensorflow as tf
from actor import ActorNetwork
from critic import CriticNetwork
from robots import gym_mountaincar, gym_pendulum


# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE =  0.01
#
ACTION_BOUND = 2.
DEVICE = '/gpu:0'

def actor_critic(epochs=1000, GAMMA = 0.99, load_file=False, render=False, temp=False, verbose=False):
    with tf.Session() as sess:
        
        
        # define objects
        # the gym environment is wrapped in a class. this way of working allows portability with other robots in the lab & makes the main very clear
        #robot = gym_pendulum(render, temp) 
        robot = gym_mountaincar(render, temp) 
        actor = ActorNetwork(sess, robot.state_dim, robot.action_dim, ACTOR_LEARNING_RATE, ACTION_BOUND, device = DEVICE)
        critic = CriticNetwork(sess, robot.state_dim, CRITIC_LEARNING_RATE, actor.get_num_trainable_vars(), device = DEVICE)
        # starting tensorflow
        sess.run(tf.global_variables_initializer())
        
        if load_file: 
            actor.recover_actor()
            critic.recover_critic()

        for i in range(epochs):
            # Reset the environment 
            state, done, step = robot.reset()
            ep_reward = 0
            
            while (not done):
                # Choose and take action, and observe reward
                action, mu, sigma = actor.predict(np.reshape(state,(1,robot.state_dim)))

                new_action = action  + 0.2*(np.random.rand(1)[0])
                action_noise = np.clip(new_action, -ACTION_BOUND, ACTION_BOUND ) 
                # print(round(action,3), round(new_action,3), round(action_noise,3),  round(mu,3), round(sigma,3))
                next_state, reward, done, step = robot.update(action_noise)
                

                # Train 
                V_minib = critic.predict(np.reshape(state, (1, robot.state_dim)))
                V_minib_next = critic.predict(np.reshape(next_state, (1, robot.state_dim)))
                if done:
                    td_target = reward
                    td_error = reward - V_minib # not - V_minib[k] ?
                else:
                    td_target = reward + GAMMA*V_minib_next
                    td_error = reward + GAMMA*V_minib_next - V_minib
                
                #critic.train(np.reshape(state, (1, robot.state_dim)), np.reshape(td_target, (1, 1)))
                critic.train(np.reshape(state, (1, robot.state_dim)), np.reshape(td_target, (1, 1)))
                actor.train(np.reshape(state,(1,robot.state_dim)), np.reshape(action, (1, 1)), np.reshape(td_error, (1, 1)) )
              
                state = next_state
                ep_reward = ep_reward + reward
                # this print is usefull for debuggin
                if verbose:
                    print(step,'action', round(action,3), 'state', round(robot.state[0],3), round(robot.state[1],3) ,'r', round(reward,3))  
               
            
            print('episode', i+1,'Steps', step,'Reward:',ep_reward,'goal achieved:', robot.goal,'Efficiency', round(100.*((robot.goal)/(i+1.)),0), '%' )
            #time.sleep(1)
            
            
               

               
        print('*************************')
        print('now we save the model')
        critic.save_critic()
        actor.save_actor()
        print('model saved succesfuly')
        print('*************************')




if __name__ == '__main__':
    actor_critic(epochs=2000, verbose = not True)       
        