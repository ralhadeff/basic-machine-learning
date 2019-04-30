# Reinforcement Learning

_work in progress_

I will implement several approaches for RL, gradually increasing in complexicity on the classic _snake_ game.

* `manual_policy` - the state space and the actions are the same as in all other implementations, but the policy mapping state &rarr; action is manually written as a dictionary (so no learning is performed). This example can be viewed as a reference for expected results with learning.  
* `dynamic_programming` - the policy and state values are initialized randomly (or arbitrarily) and the program iterates through all states to calculate the values and improve the policy until it converges on the best policy. In practice, this is usually not very useful (because the entire state space might not be accessible or feasible to loop through) and was not used for snake. However, the reference code is provided in the notebook.  
* `monte_carlo` - the policy and Q values are updated after each play (Monte Carlo sampling) using an ε-greedy explore-exploit strategy (see [video](https://www.youtube.com/watch?v=l0sFUU7vScA) of results).   
* `sarsa` - TD(0) using the [SARSA](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action) algorithm, where the policy and Q values are updated 'online'.   
* `q_learning` - TD(0) using Q-learning, where the policy and Q values are updated online but off-policy (meaning the policy used is not necessarily equal to the Q values used for updating the optimal policy). Note - in the ε-greedy case used here, this is almost identical to SARSA.  
  

<br> 

![alt text](https://github.com/ralhadeff/machine-learning-tools/blob/master/ReinforcementLearning/animations/monte_carlo.gif "RL example (Monte Carlo)")
<br>


---

**Moon landing** - experimental RL with moon landing game (inspired by [Beresheet](https://en.wikipedia.org/wiki/Beresheet))
