# Continuos Actor-Critic Implementation

A Continuos actor-critic algorithm for solving the inverted pendulum problem. The environment is provided by gym. 
This is an On-Policy algorithm.


## Requirements

- Tensorflow
- Gym
- Numpy 

## How to Run: 

```
python main.py
```


It runs the main algorithm. Both actor and critic have 2 hidden layers with 500 neurons. 
I found that the agent wasn't able to solve the problem as provided by gym, so in this version the pendulum always starts from a balanced position. 
More experiments are needed to tune it in any starting position, or maybe a more constrained initial states can be used. 
I also re tuned the reward function since it originally goes from (-16., 0.), I change this interval to (-15, 1.)

## Sources: 


- [Policy Gradient Methods, From David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf)
- [Sutton](http://incompleteideas.net/book/bookdraft2018jan1.pdf)
- [DennyBritz RL examples](https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient)