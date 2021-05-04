# Human-level control through deep reinforcement learning

### Difficulties
Reinforcement learning is unstable or even to diverge when using a neural network to represent the action-value (also known as Q) function.
- the correlations present in the sequence of observations.
- the fact that small updates toQmay significantly change the policy and therefore change the data distribution.
- the correlations between the action-values and the target values.

### Solutions
- Using a biologically inspired mechanism termed experience replay that randomizes over the data, thereby 
  removing correlations in the observation sequence and smoothing over changes in the data distribution.
- Using an iterative update that adjusts the action-values (Q) towards target values that are only periodically updated, 
  thereby reducing correlations with the target.
  

