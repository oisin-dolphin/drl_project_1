## Learning Algorithm

My first approach used the DQN algorithm. 


I used the following hyper parameters. 
* Epsilon start = 1.0
* Epsilon decay = 0.99
* Epsilon end = 0.01
* BUFFER_SIZE = int(1e5)  # replay buffer size
* BATCH_SIZE = 64  # minibatch size
* GAMMA = 0.99  # discount factor
* TAU = 1e-3  # for soft update of target parameters
* LR = 5e-4  # learning rate
* UPDATE_EVERY = 4  # how often to update the target network

I then updated this to a dueling dqn and the algorithm solved the environment in less than 500 steps.

The neural network consisted of 4 linear layers. The output is calculated as advantage action estimate (size n_action) + state value estimate (size 1). 

Layers for advantage estimate: n_states -> 64 -> relu -> 32 -> relu -> n_actions. 
Layers for value estimate: n_states -> 64 -> relu -> 32 -> relu -> 1 (shares first 3 layers with advantage estimate)


## Plot of Rewards

![plot of results](https://github.com/oisin-dolphin/drl_project_1/blob/master/plot.png)

## Ideas for future work
I would have liked to have implemented prioritised experience replay.
