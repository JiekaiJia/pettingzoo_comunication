# pettingzoo_comunication
### RLlib
**rollout**: A simulation of a policy in an environment.  
- **command line examples**  
train:`rllib train --run DQN --env CartPole-v0  # --eager [--trace] for eager execution`  
tensorboard `tensorboard --logdir=~/ray_results`  
evaluating `rllib rollout \
    ~/ray_results/default/DQN_CartPole-v0_0upjmdgr0/checkpoint_1/checkpoint-1 \
    --run DQN --env CartPole-v0 --steps 10000`  
- **configuration parameters**  
You can control the degree of parallelism used by setting the ***num_workers*** hyperparameter for most algorithms. 
  The number of GPUs the driver should use can be set via the ***num_gpus*** option. Similarly, the resource allocation 
  to workers can be controlled via ***num_cpus_per_worker***, ****num_gpus_per_worker***, and ***custom_resources_per_worker***. 
  The number of GPUs can be a fractional quantity to allocate only a fraction of a GPU. 
  For example, with DQN you can pack five trainers onto one GPU by setting num_gpus: 0.2.