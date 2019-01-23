"""
This module contains several algorithms for optimizing the multi-armed bandit choice / explore-exloit strategy
"""

import numpy as np

class BasicPlayer():
    """
    The basic algorithm for keeping track of performance of the 'bandits'
    This class is a scaffold on which the other algorithms will be constructed upon (using inheritence)
    """
    
    def __init__(self):
        # total accumulated profit (or loss)
        self.profit = 0
        
    def start(self,bandits):
        """Effectively initialize the player"""
        # number of bandits
        n = len(bandits)
        # array of estimated means for bandits
        self.means = np.zeros(n)
        # array of plays per bandit:
        # counts start from one because count is used in the denominator throughout many algorithms
        # also in optimistic initial value means is multiplied by counts at the beginning (running means)
        # the effect of starting from 1 rather than 0 is negligible after several iterations
        self.counts = np.ones(n,int)
        
    def correct_counts(self):
        """
        If user wishes to correct the counts, this method should not be called before running several iterations
        """
        self.counts-=1
    
    def update(self,bandit,outcome):
        """
        Update player after having played a bandit
        Bandit is an index
        """
        self.means[bandit] = ((self.means[bandit]*self.counts[bandit])+outcome)/(self.counts[bandit]+1)
        # add one count
        self.counts[bandit]+=1
        # accumulate profit (or loss)
        self.profit+=outcome
       
    def regret(self,bandits):
        """Returns the current regret"""
        # find the highest true mean
        u_best = max([b.mean for b in bandits])
        # total plays
        N_total = self.counts.sum()
        # calculate regret (profit vs mean profit if only the best bandit was plays all the time)
        return N_total * u_best - self.profit
    
    def log(self):
        # print profit for testing purposes
        return round(self.profit,2)
  
    def __str__(self):
        """For print-out purposes"""
        return 'Name not specified'

class RandomPlayer(BasicPlayer):
    """
    Random player picks one bandit with no preference. Used for testing purposes
    """
            
    def choose(self):
        return np.random.randint(0,len(self.means))
    
    def __str__(self):
        return 'RandomPlayer'

class EpsilonGreedy(BasicPlayer):
    """
    Epsilon greedy player chooses greedily (higher estimated mean),
    with the exception of random exploration at a chance of epsilon (typically a small number)    
    """
    
    def __init__(self,epsilon):
        """Epsilon is a hyper-parameter, typically in the range of 0.01"""
        BasicPlayer.__init__(self)
        self.epsilon = epsilon
           
    def choose(self):
        """Randomly determine whether to explore or exploit"""
        rand = np.random.rand()
        if (rand<self.epsilon):
            # explore - randomly choose any bandit
            return np.random.randint(0,len(self.means))
        else:
            # exploit - pick best bandit so far
            return np.argmax(self.means)
    
    def __str__(self):
        return 'EpsilonGreedy epsilon: '+str(self.epsilon)

class DecayingEpsilonGreedy(EpsilonGreedy):
    """
    Same as epsilon greedy, but epsilon decays with time.
    After some time the means should converge, and so epsilon greedy should stop exploring altogether
    
    There are different decay schemes, here 2 are implemented: 1/N or decay^N
    Note: 1/N tends to decay too quickly, user can therfore prolong the starting exploration
        by providing a starting point b/N
    """
    
    def __init__(self,b=None,decay=None):
        """
        User can specify b for epsilon(N)=b/N decay (b=1 for 1/N)
        OR
        decay for epsilon(N) = 1*decay**N
        """
        EpsilonGreedy.__init__(self,1)
        if (b is not None):
            self.b=b
            self.decay=None
        elif (decay is not None):
            self.decay=decay
            self.b=None
        else:
            raise ValueError('At least one input parameter must be specified for decay scheme')
            
    def choose(self):
        # update epsilon
        if (self.decay is None):
            self.epsilon = (self.b/(self.counts.sum()))
        else:
            # decay epsilon each step
            self.epsilon*=self.decay
        # after epsilon update, continue like regular EpsilonGreedy
        return EpsilonGreedy.choose(self)
    
    def __str__(self):
        string = 'DecayingEpsilonGreedy'
        if (self.decay is None):
            string+='b: '+str(self.b)
        else:
            string+='decay: '+str(self.decay) 
        return string

class OptimsiticInitialValue(BasicPlayer):
    """
    Optimistic initial value player assumes that all means are very high, then updates each iteration
    Initial value must overshoot, and with each update they become more and more realistic,
    until they converge close to the true mean
    In practice, the highest true mean will converge, and the other bandits
        will stop being explored with estimated means just below that
    """
    
    def __init__(self,optimistic_value):
        BasicPlayer.__init__(self)
        # save starting value for printout purposes
        self.optimistic_value = optimistic_value
        
    def start(self,bandits):
        BasicPlayer.start(self,bandits)
        # set starting means to optimistic value
        self.means[:]= self.optimistic_value

    def choose(self):
        # always exploit, based on overestimated means
        return np.argmax(self.means)
    
    def __str__(self):
        return 'OptimisticInitialValue:'+str(self.optimistic_value)

class UCB(BasicPlayer):
    """
    Upper confidence bound player relies on the normal distribution of the samples, 
        and is greedy towards the means including confidence intervals, thus choosing the bandit
        whose higher bound is the highest. In time the mean converges, and the bounds shrink.
        When the bounds are high, the player explores more often, and later he mostly exploits
        
    Note: UCB is sensitive to variance, so the user can provide an estimate for the expected variance
        so that the confidence intervals will be adjusted
        
    """
    
    def __init__(self,variance=1):
        BasicPlayer.__init__(self)
        # input variance
        self.variance = variance
              
    def choose(self):
        # determine the UCBs for all bandits
        # mean + confidence interval for N total plays and n plays for bandit (multiplied by variance)
        ucb = self.means + self.variance * np.sqrt((2*np.log(self.counts.sum()))/self.counts)
        return np.argmax(ucb)
        
    def __str__(self):
        string = 'UpperConfidenceBound'
        if (self.variance==1):
            return string
        else:
            return string + ' variance: ' + str(self.variance)

class ThompsonSampling(BasicPlayer):
    """
    Gaussian Thompson sampling - this algorithm generates a distribution for the predicted mean,
        randomly chooses a point from each distribution and uses this value greedily
        At the beginning, the distributions are broad, and the algorithm explores
        As the player updates, the distribution become more narrow, as confidence increases,
        and the player begins exploiting
        
    Note: TS too is sensitive to variance, and user should provide an estimated variance
    
    This code was inspired by https://github.com/lazyprogrammer/machine_learning_examples/blob/master/rl/comparing_explore_exploit_methods.py
    """

    def __init__(self,variance=1):
        BasicPlayer.__init__(self)
        self.variance = variance
    
    def choose(self):
        # pick values from mean distributions
        t = np.zeros(len(self.means))
        #for i in range(len(self.means)):
            #t[i] = np.random.normal(scale=self.variance) / np.sqrt(self.counts[i]) + self.means[i]
        t = np.random.normal(scale=self.variance,size=len(self.means)) / np.sqrt(self.counts) + self.means    
        return np.argmax(t)
  
    def __str__(self):
        string = 'GaussianThompsonSampling'
        if (self.variance==1):
            return string
        else:
            return string + ' variance: ' + str(self.variance)

class SelfRegulatingThompsonSampling(BasicPlayer):
    """
    This is an experimental derivative of ThompsonSampling that doesn't require any hyper-parameters
    
    It relies on the binary Thompson sampling (see below) and uses a beta distribution
        the outcome is normalized to 0 or 1 based on whether it is above or below the current average profit of the player
        this method basically weighs only whether a bandit is better than the current strategy or not
        it performs fairly well on a wide range of inputs, but with hyper-parameter tweaking, other algorithms perform better
        the current weakness is that after convergence, the algorithm will erode the estimated mean of the best bandit
        resulting in return to explore after many iterations
        
    I will come back to this issue and try to find a workaround sometime in the future (written Jan 2019)
    """

    def start(self,bandits):
        BasicPlayer.start(self,bandits)
        # start at 1 for beta distribution
        self.wins = np.ones(len(bandits))
        self.loses = np.ones(len(bandits))

    def correct_counts(self):
        BasicPlayer.correct_counts(self)
        self.wins-=1
        self.loses-=1
        
    def choose(self):
        samples = np.random.beta(self.wins,self.loses)
        return np.argmax(samples)

    def update(self,bandit,outcome):
        BasicPlayer.update(self,bandit,outcome)
        N = self.counts.sum()
        if outcome>self.profit/N:
            self.wins[bandit]+=1
        else:
            self.loses[bandit]+=1
    
    def __str__(self):
        return 'SelfRegulatingThompsonSampling(Experimental)'

class BinaryThompsonSampling(SelfRegulatingThompsonSampling):
    """
    Binary  Thompson sampling - same as ThompsonSampling but the input is 0 or 1
        and the distribution is beta instead of Gaussian
        
    As a minor generalization, the player will treat all positive outcomes as a win and all negative as a lose
    """
    
    def update(self,bandit,outcome):
        BasicPlayer.update(self,bandit,outcome)
        if outcome>0:
            self.wins[bandit]+=1
        else:
            self.loses[bandit]+=1
  
    def __str__(self):
        return 'BinaryThompsonSampling'

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
