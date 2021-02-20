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

Two techniques looked at here are **cross entropy** and **hill climbing**.

### Installing required packages
To install all required packages, run:
```pip install -r requirements.txt```