#Continuos Actor-Critic Implementation

A Continuos actor-critic algorithm for solving the mountain car problem. The environment is provided by gym. 
This is an On-Policy algorithm, with update rule: 


##Requirements

- Tensorflow
- Gym
- Numpy 

##Running: 

python main.py

It runs the main algorithm. Both actor and critic have 2 hidden layers with 150 neurons. 

##Sources: 


- [Policy Gradient Methods, From David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf)
- [Sutton](http://incompleteideas.net/book/bookdraft2018jan1.pdf)
- [DennyBritz RL examples](https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient)