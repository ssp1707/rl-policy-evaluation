# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT

The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States

The environment has 7 states:
* Two Terminal States: **G**: The goal state & **H**: A hole state.
* Five Transition states / Non-terminal States including  **S**: The starting state.

### Actions

The agent can take two actions:

* R: Move right.
* L: Move left.

### Transition Probabilities

The transition probabilities for each action are as follows:

* **50%** chance that the agent moves in the intended direction.
* **33.33%** chance that the agent stays in its current state.
* **16.66%** chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards

The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation
<p align="center">
<img width="600" src="https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/e7af87e7-fe73-47fa-8bea-2040b7645e44"> </p>


## POLICY EVALUATION FUNCTION

### Formula
<img width="350" src="https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/e663bd3d-fc85-41c3-9a5c-dffa57eae250">

### Program

```py
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
   	'''Initialize 1st Iteration estimates of state-value function(V) to zero'''
    prev_V = np.zeros(len(P), dtype=np.float64)

    while True:
        '''Initialize the current iteration estimates to zero'''
        V=np.zeros(len(P),dtype=np.float64)
        
        for s in range(len(P)):
        
            '''Update the value function for each state'''
            for prob,next_state,reward,done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
                
            '''Check for convergence'''
            if np.max(np.abs(prev_V-V))<theta:
                break
                
            '''Update the previous state-value function'''
            prev_V=V.copy()
        return V
```

## OUTPUT:
### Policy 1
![POLICY1](https://github.com/BHUVANESHWAR-BHUVIOP/rl-policy-evaluation/assets/94155099/e056ac3e-c5b1-4b57-8beb-0871da8ecf6d)
![SV1](https://github.com/BHUVANESHWAR-BHUVIOP/rl-policy-evaluation/assets/94155099/8991c553-4686-4cd9-9af1-e89903fd728b)
![A1](https://github.com/BHUVANESHWAR-BHUVIOP/rl-policy-evaluation/assets/94155099/6a29fa68-e35b-407c-958a-5ccb98cc57af)


### Policy 2
![POLICY2](https://github.com/BHUVANESHWAR-BHUVIOP/rl-policy-evaluation/assets/94155099/a25f1501-7569-4e04-8927-7d9fca399c01)
![SV2](https://github.com/BHUVANESHWAR-BHUVIOP/rl-policy-evaluation/assets/94155099/9755afb0-113a-4ded-9188-fea897a03208)
![A2](https://github.com/BHUVANESHWAR-BHUVIOP/rl-policy-evaluation/assets/94155099/106c6e18-e1c3-497f-940d-897eec173002)


### Comparison
![COMP](https://github.com/BHUVANESHWAR-BHUVIOP/rl-policy-evaluation/assets/94155099/576d1cc5-3edd-407c-99e3-56696241d1be)


### Conclusion

  
![CON](https://github.com/BHUVANESHWAR-BHUVIOP/rl-policy-evaluation/assets/94155099/501406a7-b76c-4903-97d3-2054d7d78c13)



## RESULT:
Thus, a Python program is developed to evaluate the given policy.
![image](https://github.com/AavulaTharun/rl-value-iteration/assets/93427201/7467f33f-47ed-4b2b-8bc2-0419bd7b3755)


## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.

