# Holt Spalding
# Character-centric Pacman Cropping Experiment
# This experiment compares the performance of a DQN agent
# trained on a uncropped greyscale representation of the 
# OpenAI MsPacman environment with that of a DQN agent
# trained on a smaller, Pacman-centric greyscale representation
# of the environment. Details of the experiment can be found
# in the associated paper.

#To run experiment 1 detailed in the paper, run python PacmanDQNExp.py -exp 1
#To run experiment 2 detailed in the paper, run python PacmanDQNExp.py -exp 2

import cv2
import gym
import sys
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from enum import Enum

class cropped(Enum):
    cropped = 1
    uncropped = 2
class colortransform(Enum):
    greyscale = 1
    binary = 2

# Class controlling MsPacman agent
class PacmanDQN():
    def __init__(self, frame_skip = 4, 
                      num_episodes = 200, 
                      gamma = 0.99, 
                      pretraining_exploration = 10000, 
                      exploration_steps = 1000, 
                      final_epsilon = 0.1, 
                      replay_memory_max_size =  60000, 
                      minibatch_size = 32, 
                      crop_mode = cropped.uncropped, 
                      colortransform = colortransform.greyscale):
       
        self.gamma = gamma                                      #discount factor
        self.frame_skip = frame_skip                            #frames skipped before action is selected
        self.num_episodes = num_episodes                        #episodes of training
        self.pretraining_exploration = pretraining_exploration  #number of time steps run before training starts
        self.exploration_steps = exploration_steps              #controls the rate at which epsilon anneals
        self.final_epsilon = final_epsilon                      #minimum value of epsilon
        self.replay_memory_max_size = replay_memory_max_size    #size of experience replay buffer
        self.minibatch_size = minibatch_size                    #size of replay batches
        self.replay_memory = deque()                        
        self.time_step = 0                                      #time steps
        self.epsilon = 1.0                                      #epsilon
        self.init(crop_mode)

    #initializes DQN 
    def init(self, crop_mode = cropped.uncropped):
        # instantiate input layer
        if crop_mode == cropped.uncropped:
            self.state_input = tf.placeholder("float", [None, 160, 160, 4])
        elif crop_mode == cropped.cropped:
            self.state_input = tf.placeholder("float", [None, 40, 40, 4])

        #instantiate convolutional layers
        conv1_weights = self.weight_variable([8, 8, 4, 32])
        conv1_bias = self.bias_variable([32])

        conv2_weights = self.weight_variable([4, 4, 32, 64])
        conv2_bias = self.bias_variable([64])

        conv3_weights = self.weight_variable([3, 3, 64, 64])
        conv3_bias = self.bias_variable([64])

        #instantiate fully connected layers
        if crop_mode == cropped.uncropped:
            fc1_weights = self.weight_variable([6400, 512])
        elif crop_mode == cropped.cropped:
            fc1_weights = self.weight_variable([576, 512])
        fc1_bias = self.bias_variable([512])

        fc2_weights = self.weight_variable([512, 9])
        fc2_bias = self.bias_variable([9])

        # instantiate hidden layers
        hidden_conv1 = tf.nn.relu(self.conv2d(self.state_input, conv1_weights, 4) + conv1_bias)
        hidden_pool1 = self.max_pool_2x2(hidden_conv1)
        hidden_conv2 = tf.nn.relu(self.conv2d(hidden_pool1, conv2_weights, 2) + conv2_bias)
        hidden_conv3 = tf.nn.relu(self.conv2d(hidden_conv2, conv3_weights, 1) + conv3_bias)
        if crop_mode == cropped.uncropped:
            hidden_conv3_flat = tf.reshape(hidden_conv3, [-1, 6400])
        elif crop_mode == cropped.cropped:
            hidden_conv3_flat = tf.reshape(hidden_conv3, [-1, 576])
        fc1_hidden = tf.nn.relu(tf.matmul(hidden_conv3_flat, fc1_weights) + fc1_bias)

        # Q Value layer
        self.Q_value = tf.matmul(fc1_hidden, fc2_weights) + fc2_bias

        self.action_input = tf.placeholder("float", [None, 9])
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.train_step = tf.train.AdamOptimizer(1e-5).minimize(self.cost)

        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    ##################################################################################################
    # the below all act in service to the initialization of our DQN
    def set_init_state(self, observation):
        self.current_state = np.stack((observation, observation, observation, observation), axis = 2)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial) 
    
    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = 'SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    #################################################################################################


    # chooses action by means of epsilon-greedy policy
    # (0 = center, 1 = up, 2 = right, 3 = left, 4 = down, 5 = upper-right, 6 = upper-left, 7 = lower-right, 8 = lower-left)
    def choose_action(self):
        #get Q_value
        Q_value = self.Q_value.eval(feed_dict ={ self.state_input:[self.current_state]  })[0]
        action = np.zeros(9)
        action_index = 0

        #choose action depending on epsilon
        if self.time_step % self.frame_skip == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(9)
            else:
                action_index = np.argmax(Q_value)

        #dynamically reduce epsilon
        if self.epsilon > self.final_epsilon and self.time_step > self.pretraining_exploration:
            self.epsilon -= (self.epsilon - self.final_epsilon) /  self.exploration_steps

        return action_index

    #trains Q network
    def train(self):
        #get random minibatch from experience replay
        minibatch = random.sample(self.replay_memory,self.minibatch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch] 
        y_batch = []
        Q_Value_batch = self.Q_value.eval(feed_dict={self.state_input:nextState_batch})
        for i in range(0,self.minibatch_size):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.gamma * np.max(Q_Value_batch[i]))
        self.train_step.run(feed_dict={
            self.y_input : y_batch,
            self.action_input : action_batch,
            self.state_input : state_batch
            })

    #trains DQN based on experience replay
    def update_curr_state(self,observation,action,reward,terminal_state):
        #save states into replay memroy
        new_state = np.append(observation,self.current_state[:,:,1:],axis = 2)
        self.replay_memory.append((self.current_state,action,reward,new_state,terminal_state))
        if len(self.replay_memory) > self.replay_memory_max_size:
            self.replay_memory.popleft()
        #train if you've already taken random actions to 
        # bootstrap exploration
        if self.time_step > self.pretraining_exploration:
            self.train()
        self.current_state = new_state
        self.time_step += 1


#sets up learning environment
def set_initial_observation(env, agent, crop_mode = cropped.uncropped, filename = "images/greyscale.png"):
    observation = env.reset()
    observation = observation[0:180]
    observation = cv2.cvtColor(cv2.resize(observation, (160, 160)), cv2.COLOR_BGR2GRAY)
    if crop_mode == cropped.cropped:
        observation = observation[70:110,60:100]
    agent.set_init_state(observation)
    cv2.imwrite(filename,observation)

#resets learning environment once an episode terminates
def reset_env(env, agent,crop_mode = cropped.uncropped):
    observation = env.reset()
    observation = observation[0:180]
    observation = cv2.cvtColor(cv2.resize(observation, (160, 160)), cv2.COLOR_BGR2GRAY)
    if crop_mode == cropped.cropped:
        observation = observation[70:110,60:100]
    agent.set_init_state(observation)

#preprocesses frame from the game depending on crop_mode
def preprocess(observation, crop_mode = cropped.uncropped, colortransform = colortransform.greyscale, y = 70, x = 60):
    if crop_mode == cropped.uncropped and colortransform == colortransform.greyscale:
        observation = cv2.cvtColor(cv2.resize(observation[0:180], (160, 160)), cv2.COLOR_BGR2GRAY)
        return np.reshape(observation, (160, 160, 1))
    elif crop_mode == cropped.cropped and colortransform == colortransform.greyscale:
        observation = observation[0:180]
        observation = cv2.cvtColor(cv2.resize(observation, (160, 160)), cv2.COLOR_BGR2GRAY)
        observation = observation[int(y)-20:(int(y)+20),int(x)-20:(int(x)+20)]
        return np.reshape(observation, (40, 40, 1))

def get_action(action):
    a = np.zeros(9)
    a[action] = 1
    return a
#retrieves MsPacman's position for spotlight mode
def get_pacman_position(frame):
    pacman_color = np.array([210,164,74])
    mask = cv2.inRange(frame, pacman_color, pacman_color)  
    coord=cv2.findNonZero(mask)
    try:
        return np.mean(coord, axis=0)[0]
    except:
        print("There is an issue determining MsPacman's position")
        return [-1, -1]

def crop_helper(x):
    try:
        if x < 20:
            return 20
        if x > 140:
            return 140
        else:
            return x
    except:
        print("worng input")


#uncropped, frameskip 4, greyscale, 
def Experiment1():
    print("Beginning Experiment 1...")
    env = gym.make('MsPacman-v0')
    agent = PacmanDQN(frame_skip = 4, crop_mode = cropped.uncropped, colortransform = colortransform.greyscale, num_episodes=200,  pretraining_exploration=10000)
    set_initial_observation(env,agent,cropped.uncropped,"images/exp1.png")
    #pretraining observation
    print("Beginning pretraining observation...")
    while agent.time_step <= agent.pretraining_exploration:
        action = agent.choose_action()
        observation,reward,terminal_state,lives = env.step(action)
        observation = preprocess(observation, crop_mode = cropped.uncropped, colortransform = colortransform.greyscale, y=0,x=0)
        agent.update_curr_state(observation,get_action(action),reward,terminal_state)
        if terminal_state == True:
            reset_env(env,agent,cropped.uncropped)
    print("Beginning training...")
    cumulative_reward = [0,0,0]
    total_rewards = []
    curr_ep = 1
    while curr_ep <= agent.num_episodes:
        action = agent.choose_action()
        observation,reward,terminal_state,lives = env.step(action)
        observation = preprocess(observation, crop_mode = cropped.uncropped, colortransform = colortransform.greyscale, y=0,x=0)
        agent.update_curr_state(observation,get_action(action),reward,terminal_state)
        cumulative_reward[lives.get("ale.lives") - 1] += reward
        if terminal_state == True:
            print("Episode " + str(curr_ep) + " Complete")
            print("Epsilon: " + str(agent.epsilon))
            reset_env(env,agent,cropped.uncropped)
            total_rewards.append(pd.Series(cumulative_reward).mean())
            curr_ep += 1
            cumulative_reward = [0,0,0]
    plt.plot(range(len(total_rewards)), total_rewards)
    plt.title('Experiment 1 (Uncropped, Greyscale, Frameskip of 4)')
    plt.xlabel('Number Of Episodes')
    plt.ylabel('Average Score Of Three Deaths')
    plt.savefig('images/Experiment1.png')
    plt.close()
    print(total_rewards)


#cropped/spotlight, frameskip 4, greyscale, 
def Experiment2():
    print("Beginning Experiment 2...")
    env = gym.make('MsPacman-v0')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('images/Experiment2.avi',fourcc, 60, (40,40))
    agent = PacmanDQN(frame_skip = 4, crop_mode = cropped.cropped, colortransform = colortransform.greyscale, num_episodes=200,  pretraining_exploration=10000)
    set_initial_observation(env,agent,cropped.cropped,"images/exp2.png")
    #pretraining observation
    print("Beginning pretraining observation...")
    while agent.time_step <= agent.pretraining_exploration:
        action = agent.choose_action()
        observation,reward,terminal_state,lives = env.step(action)
        coord = get_pacman_position(observation)
        observation = preprocess(observation, crop_mode = cropped.cropped, colortransform = colortransform.greyscale, y=crop_helper(coord[1]), x=crop_helper(coord[0]))
        agent.update_curr_state(observation,get_action(action),reward,terminal_state)
        if terminal_state == True:
            reset_env(env,agent,cropped.cropped)
        out.write(cv2.cvtColor(observation, cv2.COLOR_GRAY2BGR))
    print("Beginning training...")
    cumulative_reward = [0,0,0]
    total_rewards = []
    curr_ep = 1
    while curr_ep <= agent.num_episodes:
        action = agent.choose_action()
        observation,reward,terminal_state,lives = env.step(action)
        coord = get_pacman_position(observation)
        if -1 in coord:
            terminal_state == True
        else:
            observation = preprocess(observation, crop_mode = cropped.cropped, colortransform = colortransform.greyscale, y=crop_helper(coord[1]), x=crop_helper(coord[0]))
            agent.update_curr_state(observation,get_action(action),reward,terminal_state)
            cumulative_reward[lives.get("ale.lives") - 1] += reward
            out.write(cv2.cvtColor(observation, cv2.COLOR_GRAY2BGR))
        if terminal_state == True:
            print("Episode " + str(curr_ep) + " Complete")
            print("Epsilon: " + str(agent.epsilon))
            reset_env(env,agent,cropped.cropped)
            total_rewards.append(pd.Series(cumulative_reward).mean())
            curr_ep += 1
            cumulative_reward = [0,0,0]
    plt.plot(range(len(total_rewards)), total_rewards)
    plt.title('Experiment 2 (Cropped, Greyscale, Frameskip of 4)')
    plt.xlabel('Number Of Episodes')
    plt.ylabel('Average Score Of Three Deaths')
    plt.savefig('images/Experiment2.png')
    plt.close()
    print(total_rewards)
    out.release()


#run experiments here
def main():
    parser = argparse.ArgumentParser(description='Determine which experiment to run.')
    parser.add_argument("-exp", dest="exp")
    args = parser.parse_args()
    if args.exp == "1":
        Experiment1()
    elif args.exp == "2":
        Experiment2()
    else:
        print("Usage: python PacmanDQNExp.py -exp <1 or 2>")

if __name__ == "__main__":
    main()
