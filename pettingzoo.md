# Peettingzoo Code Reading

### Senario class in simply_sread.py. 
- make_world()  
Sets communication dimension dim_c to 2, agent to silent.  
- reset_world()  
Initializes the agents' and landmarks' state and color. agent state: position p_pos, velocity p_vel and communication c.(dim=2)  
agent action: 
- physical action u [0,0], [0,1], [1,0], [-1,0], [0,-1]
- communication action c [1,0], [0,1]
### World class in core.py
- step()  
Calculates the agents' position, velocity and communication state (state.c = action.c + noise)
### SimpleEnv class in simple_env.py
- step()  
Executes the world.step() and updates the reward at that time.

### Basic concepts
- reward: reward = global reward * (1 - ratio) + agent reward * ratio 
    - global reward: minus the minimun distance between landmarks and agents.
    - agent reward: minus the amount of agents' collisions.
- observation: [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]
- action: [no_action, move_left, move_right, move_down, move_up]

### RLlib
- rollout_worker.py
- sampler.py (by default SyncSampler)
- dynamic_tf_policy.py
- 
