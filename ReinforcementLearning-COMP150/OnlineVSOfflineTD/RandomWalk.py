#Holt Spalding
# Random Walk Experiment
# This experiment compares the performance of online and offline TD agents on 
# a random walk task. Details on the experiment are provided in the corresponding 
# paper. 


import numpy as np
import matplotlib.pyplot as plt
import math
import random


#This class initializes the experiment described in the corresponding paper.
#See the paper to understand the following random walk task
class RandomWalk:
    def __init__(self, num_states = 19, method = "online",  gamma = 1):
        self.num_states = num_states                 #number of states in task
        self.method = method                         #can be 'online' or 'offline'
        self.gamma = gamma                           #discount factor
        ns = num_states + 1
        self.states = np.arange(1, ns)               #id num for each state, terminal states excluded
        self.start = math.ceil(ns / 2)               #start state is always in the middle
        self.terminal_states = [0,ns]                #terminal states of the random walk
        self.V_star = np.arange(-ns, ns+2,2) / ns    #this represents the true value of each state according to the Bellman equation
        self.V_star[0] = self.V_star[-1] = 0


    #runs an episode on the random walk task and updates value function saved in 'value'
    def td_episode(self, value, alpha, n):
        curr_state = self.start                   #state at current time step
        gamma      = self.gamma                   #discount factor
        m          = self.method                  #online or offline learning method
        if m == 'offline':
            delta = np.zeros(self.num_states + 2) #saves value updates for offline learning

        t = 1                                    #time step
        T = float('inf')                         #time step at terminal state
        tau = -1                                 #time step whose state's estimates are updated when running online TD 
       
        visited_states = [curr_state]            #keeps track of states visited during episode
        rewards = [0]                            #keeps track of rewards observed during episode

    
        while tau != T - 1:
            #Randomly choose actions and record states and rewards observed
            if t < T:
                #Random action is chosen, 0 represents a move left, 1 represents a move right
                if random.getrandbits(1) == 0:
                    next_state = curr_state - 1
                else:
                    next_state = curr_state + 1

                reward = 0
                if next_state == (self.num_states + 1):
                    reward = 1
                elif next_state == 0:
                    reward = -1                    

                #reward and state observed is recorded
                rewards.append(reward)
                visited_states.append(next_state)

                #If terminal state is reached, cease taking action
                if next_state in self.terminal_states:
                    T = t
        
            tau = t - n
            #If you've completed over n time steps, update state-value function
            if tau >= 0:
                returns = 0.0
                for i in range(tau + 1, min(T, tau + n) + 1):
                    returns += pow(gamma, i - tau - 1) * rewards[i]
                if tau + n <= T:
                    returns += pow(gamma, n) * value[visited_states[(tau + n)]]
                #update state values depending on TD method
                if not visited_states[tau] in self.terminal_states:
                    if m == 'online':
                        value[visited_states[tau]] += alpha * (returns - value[visited_states[tau]])
                    elif m == 'offline':
                        delta[visited_states[tau]] += alpha * (returns - value[visited_states[tau]])


            curr_state = next_state
            t += 1                     #increment timestep
        if m == 'offline':
            value += delta



#calculates RMS error
def RMSE(prediction, model):
    return np.sqrt(np.sum(np.power(prediction - model.V_star, 2)) / model.num_states)

#runs experiment described in corresponding paper and saves graphs
def experiment(eps, runs):
    alphas = np.arange(0, 1.1, 0.1)
    nsteps     = np.power(2, np.arange(0, 10))
    model_online = RandomWalk(method='online')
    model_offline = RandomWalk(method='offline')

    RMS_matrix_online = np.zeros((len(nsteps), len(alphas)))
    RMS_matrix_offline = np.zeros((len(nsteps), len(alphas)))

    for r in range(0,runs):
        for sidx, stepval in zip(range(len(nsteps)), nsteps):
            for aidx, alphaval in zip(range(len(alphas)), alphas):
                values_online = np.zeros(model_online.num_states+2)
                values_offline = np.zeros(model_offline.num_states+2)
                for ep in range(0,eps):
                    model_online.td_episode(values_online,alphaval, stepval)
                    model_offline.td_episode(values_offline, alphaval, stepval)

                    RMS_matrix_online[sidx,aidx] += RMSE(values_online, model_online)
                    RMS_matrix_offline[sidx,aidx] += RMSE(values_offline, model_offline)
        print("Now on run " + str(r))
    RMS_matrix_online /= runs * eps
    RMS_matrix_offline /= runs * eps

    for i in range(0, len(nsteps)):
        plt.plot(alphas, RMS_matrix_online[i, :], label='n = %d' % (nsteps[i]))
    plt.ylim([0.25, 1])
    plt.legend()
    plt.title('Online TD Performance On Random Walk Task')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Average RMS error over 19 states and first 10 episodes')
    plt.savefig('Online.png')
    plt.close()

    for i in range(0, len(nsteps)):
        plt.plot(alphas, RMS_matrix_offline[i, :], label='n = %d' % (nsteps[i]))
    plt.ylim([0.25, 1])
    plt.legend()
    plt.title('Offline TD Performance On Random Walk Task')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Average RMS error over 19 states and first 10 episodes')
    plt.savefig('Offline.png')
    plt.close()



if __name__ == '__main__':
    experiment(eps = 10, runs = 200) 