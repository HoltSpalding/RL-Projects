import numpy as np
from random import randint
import matplotlib.pyplot as plt

# totalreward = 0.0; #total reward over all time steps
# avgreward = []   #avg reward at each time step in list
# optactcount = 0.0 # number of times optimal action is taken
# peroptact = [] #percentage of times optimal action chosen at each time step

#asm = 0 => sample average
#asm = 1 => alpha
class BanditTestbed:
    def __init__(self, k, steps, asm, epsilon, stationary, alpha):
        self.q = np.zeros(k)
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.steps = steps
        self.asm = asm
        self.epsilon = epsilon
        self.stationary = stationary
        self.alpha = alpha
        self.totalreward = 0.0
        self.avgreward = []
        self.optactcount = 0.0
        self.peroptact = []

    def experiment(self):
        for i in range(1,self.steps+1):    
            A = np.random.choice([np.random.choice(np.flatnonzero(self.Q == self.Q.max())),randint(0,9)],  p=[1-self.epsilon, self.epsilon])
            R = np.random.normal(loc=self.q[A])

            self.totalreward += R;
            self.avgreward = self.avgreward + [self.totalreward/i]
            self.optactcount += (np.argmax(self.q) == A)*100
            self.peroptact = self.peroptact + [self.optactcount/i]


            self.N[A] = self.N[A] + 1
            if (self.asm == 0):
                self.Q[A] = self.Q[A] + (1/self.N[A]) * (R - self.Q[A])
            elif (self.asm == 1):
                self.Q[A] = self.Q[A] + self.alpha * (R - self.Q[A])
            else:
                print("Sampling method has not been implemented yet")


            #adding noise if nonstationary
            if (self.stationary == False):
                for j in range(0,9):
                    noise = np.random.normal(loc=0,scale=0.01)
                    self.q[j] = self.q[j] + noise



def main():
    sa = BanditTestbed(k=10, steps=10000, asm=0, epsilon=0.1, stationary=False, alpha=0.1)
    erwa = BanditTestbed(k=10, steps=10000, asm=1, epsilon=0.1, stationary=False, alpha=0.1)

    sa.experiment();
    erwa.experiment();

    plt.plot(sa.avgreward, c = "blue", label="sample average");
    plt.plot(erwa.avgreward, c = "red", label="exponential recency weighted average");
    plt.xlabel("steps", fontsize=20)
    plt.ylabel("Average Reward", fontsize=20)
    plt.legend()
    outfile = "output/avgreward.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()

    plt.plot(sa.peroptact, c = "blue", label="sample average");
    plt.plot(erwa.peroptact, c = "red", label="exponential recency weighted average");
    plt.xlabel("steps", fontsize=20)
    plt.ylabel("%Optimal Action", fontsize=20)
    plt.legend()
    outfile = "output/peroptact.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
