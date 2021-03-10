# Policy Based Methods

This code base contains examples of implementations of **policy based methods**, a set of reinforcement learning algorithms
which allows derivation of an **optimal policy** for an agent, without having to derive an **optimal value function** first.
(The knowledge shared here is taken from the Udacity Reinforcement Learning Nanodegree.)

There are three reasons why **policy based methods** are useful:
1. **Simplicity**: Policy based methods get to the problem at hand directly, without having to store extra data as is the
   case for **value based methods**
1. **Stochastic Policies**: Policy based methods can learn true stochastic values. Not the case for **value based methods**
   where action selection introduces randomness
1. **Continuous action space**: Policy based methods are well-equipped to deal with continuous action spaces

The techniques looked at here are **cross entropy**, **hill climbing** and the **deep deterministic policy gradient**.

### Installing required packages
To install all required packages, run:
```pip install -r requirements.txt```

### Usage
A single script can be executed from the command line to run different implementations of policy gradient methods. The 
help print out showing the usage can be done running the following command:
```commandline
python runner.py -h
```
The following shows an example of running the DDPG algorithm with 2000 episodes, a 1000 time steps and, a flag indicating
whether to plot the results upon training completion:
```commandline
python runner.py ddpg --episodes 2000 --timesteps 1000 --plot true
```