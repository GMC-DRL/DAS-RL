# DAS-RL：Dynamic Algorithm Selection via Reinforcement Learning

## Testing.py (main entrance)

### Parameter setting

Edit line 454~493 to modify the setting of algorithm:

+ VectorEnv: The environment structure, can be **env.DummyVectorEnv** for serial environment
or **env.SubprocVectorEnv** for parallel environment (only for Linux).
+ problem: The problem to be solve, can be a str (e.g. "Schwefel") or a list of str (e.g. ["Schwefel"] and ["Schwefel", "Bent_cigar"])
+ subproblems: The list of suubproblems of Composition and Hybrid problem (e.g. ['Rastrigin', 'Happycat', 'Ackley', 'Discus', 'Rosenbrock'])
+ sublength: The partition of Hybrid problem dimensions (e.g. [0.1, 0.2, 0.2, 0.2, 0.3] with sum = 1)
+ indicated_dataset: Indicate a mixed dataset with a problem collection in env/training_dataset.py, it will overwrite *problem*, *subproblems* and *sublength* above. **None** for not to indicate.
+ shifted: A Bool type indicates whether to shift or not
+ rotated: A Bool type indicates whether to rotate or not
+ Train_set: The size of training set
+ Test_set: The size of testing set
+ rl: A str indicates the using of PPO agent (**"PPO"**) or DQN agent (**"DQN"**)
+ buf_size: The size of replay memory for DQN
+ rep_size: The size of replay sample size for DQN
+ dim: The dimension of problems
+ batch_size: The size of problem batches
+ MaxFEs: The maximum number of function evaluations
+ period: The period for switching optimization algorithms
+ Epoch: The number of training epoches
+ epsilon: The epsilon for epsilon-greedy for DQN
+ epsilon_decay: The decay of epsilon
+ lr: The learning rate for DQN or actor of PPO
+ critic_lr: The learning rate for critic of PPO
+ lr_decay: The decay rate of learning rate ofr lr and critic_lr
+ sample_times: The number of samples in state generation
+ sample_size: The size of sampled population in state generation, -1 for "be the same of current population"
+ k_epoch: The number of training of PPO in each trajectory segment
+ testing_repeat: The number of runs for each test
+ testing_internal: The internal between two tests. Minus number for disable testing.
+ testing_seeds: The random seed of testing
+ data_gen_seed: The random seed of dataset generation
+ optimizers: The name of backboned algorithms (e.g.['NL_SHADE_RSP', 'MadDE', 'JDE21'])
+ terminal_error: The error value of costs indicating the end of optimization
+ state_dict: The loaded model of agent, **None** to disable it
+ device: The device of network (e.g. 'cuda:0')


### Run

Python Testing.py

The tensorboard log will be stored in log/ (need manual creation) with a file name contains the used agent and runtime (e.g. **PPO-YYYYMMDDTHHmmSS**)
The saved models are in /save_policy_PPO or /save_policy_DQN with runtime file name (e.g. **YYYYMMDDTHHmmSS**)

### ploter.py & optimizer_dist.py

Plotting some pictures in paper. *optimizer_dist* plots the distribution of selected actions along optimization and *ploter* plots the picture of 2 dimension problems

