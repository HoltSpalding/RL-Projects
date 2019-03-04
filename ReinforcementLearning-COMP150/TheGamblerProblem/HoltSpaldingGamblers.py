#################################################
# Author: Holt Spalding
# Gambler's Problem: Sutton & Barto Exercise 4.9
# Date: 10/2/18
##################################################

import matplotlib.pyplot as plt
import numpy as np


##########################
# Gambler experiment class
###########################
class GamblerExp:
    def __init__(self, num_states, ph, gamma, theta):
        self.V = np.zeros(num_states + 1)  #value estimates of each state
        self.r = np.zeros(num_states + 1)  #reward space
        self.r[num_states] = 1
        self.pi  = np.zeros(num_states + 1) #policy of each state
        self.num_states = num_states        #number of states
        self.ph = ph                        #probability of heads
        self.gamma = gamma                  #discount factor
        self.theta = theta                  #sweep threshold


    ######################
    # Runs actual experiment to learn optimal policy
    #####################
    def experiment(self):
        delta = 1000  
        iters = 0
        ###########################
        # Basic value iteration algorithm
        ##########################
        while(delta >= self.theta):
            delta = 0
            for s in range(1,self.num_states):
                v = self.V[s]
                self.bellman_update(s)
                delta = max(delta, abs(v - self.V[s]))
            iters += 1

    ###################################################
    #  The below code is all just for plotting purposes
    ##################################################
            if(iters == 1):
                plt.plot(self.V[0:self.num_states-1], c = "black", label="sweep 1")
            if(iters == 2):
                plt.plot(self.V[0:self.num_states-1], c = "blue", label="sweep 2")
            if(iters == 3):
                plt.plot(self.V[0:self.num_states-1], c = "green", label="sweep 3")

        
        plt.plot(self.V[0:self.num_states-1], c = "red", label = "sweep " + str(iters))
        plt.legend()
        plt.suptitle('Value Estimates With ' + str(self.ph) + ' Probability Of Heads')
        plt.xlabel('Capital')
        plt.ylabel('Value estimates')
        outfile = "gpvalest" + str(self.ph) + ".png"
        plt.savefig(outfile, bbox_inches="tight")
        plt.show()
        plt.close()

        plt.plot(self.pi)
        plt.suptitle('Optimal Policy With ' + str(self.ph) + ' Probability Of Heads')
        plt.xlabel('Capital')
        plt.ylabel('Final policy (stake)')
        outfile = "gpfinpol" + str(self.ph) + ".png"
        plt.savefig(outfile, bbox_inches="tight")
        plt.show()


    ###################################################################
    # Value function update using modified Bellman optimality equation
    ###################################################################
    def bellman_update(self, capital):
        curr_q_value = 0

        for stake in range(1, min(capital, self.num_states - capital) + 1): #analyzing every possible action
            reward_if_win = self.r[capital + stake] + self.gamma * self.V[capital + stake]  #reward in a winning scenario
            reward_if_lose = self.r[capital - stake] + self.gamma * self.V[capital - stake] #reward in a losing scenario

            q_value = self.ph*reward_if_win + (1 - self.ph)*reward_if_lose   #expected reward given a state and action

            #updating our value function and policy if action maximizes expected reward
            if(q_value > curr_q_value):
                self.pi[capital] = stake
                self.V[capital] = q_value



def main():
    # ph = 0.55 experiment
    p1 = GamblerExp(num_states = 100, ph = 0.55, gamma = 1, theta = 1e-50)
    p1.experiment()
    # ph - 0.25 experiment 
    p2 = GamblerExp(num_states = 100, ph = 0.25, gamma = 1, theta = 1e-50)
    p2.experiment()


if __name__ == "__main__":
    main()
