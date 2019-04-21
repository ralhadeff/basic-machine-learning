# Reinforcement Learning

_work in progress_

I will implement several approaches for RL, gradually increasing in complexicity on the classic _snake_ game.

* `manual_policy` - the state space and the actions are the same as in all other implementations, but the policy mapping state -> action is manually written as a dictionary (so no learning is performed). This example can be viewed as a reference for expected results with learning.  
* `dynamic_programming` - the policy and state values are initialized randomly (or arbitrarily) and the program iterates through all states to calculate the values and improve the policy until it converges on the best policy. In practice, this is usually not very useful (because the entire state space might not be accessible or feasible to loop through) and was not used for snake. However, the reference code is provided in the notebook.  
* `X` - y.  


  
    
      
<br>      
<br>

**Moon landing** - experimental RL with moon landing game (inspired by [Beresheet](https://en.wikipedia.org/wiki/Beresheet))
