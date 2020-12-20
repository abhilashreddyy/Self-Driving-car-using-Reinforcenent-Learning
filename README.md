# Self Driving Car using Reinforcement Learning

## Introduction

- This repository consists of Implementations for manuovering a car on google maps roads
- I have implemented this using 2 Reinforcement Learning algorithms
    1. Self Driving car using Q-Learning ([github](https://github.com/abhilashreddyy/Self-Driving-car-using-Reinforcenent-Learning/tree/main/Q-Learning))
    2. Self Driving car using Twin Delayed Deep Determinsitic Network (TD3) ([github](https://github.com/abhilashreddyy/Self-Driving-car-using-Reinforcenent-Learning/tree/main/TD3_algorithm))

## Reinforcement Learning:
-  Reinforcement learning is the training of machine learning models to make a sequence of decisions. The agent learns to achieve a goal in an uncertain, potentially complex environment. In reinforcement learning, an artificial intelligence faces a game-like situation. The computer employs trial and error to come up with a solution to the problem. To get the machine to do what the programmer wants, the artificial intelligence gets either rewards or penalties for the actions it performs. Its goal is to maximize the total reward.
- Although the designer sets the reward policy–that is, the rules of the game–he gives the model no hints or suggestions for how to solve the game. It’s up to the model to figure out how to perform the task to maximize the reward, starting from totally random trials and finishing with sophisticated tactics and superhuman skills. By leveraging the power of search and many trials, reinforcement learning is currently the most effective way to hint machine’s creativity. In contrast to human beings, artificial intelligence can gather experience from thousands of parallel gameplays if a reinforcement learning algorithm is run on a sufficiently powerful computer infrastructure.

## Briefing The current application:
- In the current application we are using two different algorithms to make a car manuover on google map roads roads. Basically these two algorithms use Bellmanfords equation as a core concept and deep learning for training a car.
- For brief understanding. Let us look a the environment

![environ](images/citymap.png)
- Observe there are roads on the map. The car is supposed to travel from a source to destination and need to learn to travel on roads. As soon as the destination is reached, the source and destination are interchanged.
- We generally punish the network if it goes away from destination or travels on space other than road
- You can see a demo of this aplication [here](https://www.youtube.com/watch?v=Gj1HzlnH-vc).

Here are the links to Home of two projects:

__NOTE__ : Go trough Q-learning before visiting these links for better understanding

- Visit here for further understanding of the project

1. Self Driving car using Q-Learning ([github](https://github.com/abhilashreddyy/Self-Driving-car-using-Reinforcenent-Learning/tree/main/Q-Learning))
2. Self Driving car using Twin Delayed Deep Determinsitic Network (TD3) ([github](https://github.com/abhilashreddyy/Self-Driving-car-using-Reinforcenent-Learning/tree/main/TD3_algorithm))
