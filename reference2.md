# Learning to communicate with deep multi-agent reinforcement learning
<https://arxiv.org/pdf/1605.06676.pdf>
## 1. Introduction
+ **What kind of problem are we handling with?**  
We consider the task as fully cooperative, partially observable, sequential multi-agent decision-making problems.
+ **What is the main design principle?**  
centralised learning but decentralised execution.
+ **How we address the tasks?**
    1. **Reinforced Inter-Agent Learning (RIAL)**.  
        This approach has the structure of deep Q-learning with a recurrent network. It has 2 different 
        variants, one is each agent learn its own network, treating the other agents as part of the 
        environment. Another is every agent shares a single network.
    2. **Differentiable Inter-Agent Learning (DIAL)**.  
        This approach is end-to-end trainable within an agent, thereby communication actions are 
        connections between agents. With communication the gradients will be not only transmitted within 
        an agent, but also transmitted across the agents.
## 2. Related work
## 3. Background
+ **Deep Q-Network(DQN)**  
    DQN uses a neural network $Q(s,u;\theta)$, where $s$, $u$ and $\theta$ is respectively 
    state, action and parameter of neural network. The loss function of DQN is as followed:
    $$\mathcal(L)_i(\theta_i) = \mathbb(E)_{s,u,r,s'}\[(y_i^{DQN} - $Q(s,u;\theta_i))^2\]$$  
    $y_i^{DQN}$ is the target value computed by a frozen network, which stop updating for a 
    few iterations. The optimization strategy is to minimize the loss function.
+ **Dependent DQN**   
    This is a network extending to multi-agent system, that is, each agent separately learning its
    own network. However, this kind of setting may lead to convergence problems.
+ **Deep Recurrent Q-Network**     
    Compared to the above 2 networks, this network is introduced in partially observable environment.
    Deep Recurrent Q-Network can maintain an internal state and aggregate observations over time.
## 4. Settings  

## 5. Models
## 6. Experiments
## 7. Conclusion

## 8. Questions
- What is the communication action exactly?
- What do the agents exactly exchange? 
- How should we define the partial observation?
- How can we define the communication oder? 
- DIAL push the gradients through communication channel.


