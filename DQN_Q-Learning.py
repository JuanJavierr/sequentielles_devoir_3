import random
import math
import json

# Class definition for Deep Q Network (DQN)
class DQN:
    """
        Initialize the DQN object with the following parameters:
        - state_dim: The dimensionality of the state space.
        - action_dim: The dimensionality of the action space.
        - hidden_dims: A list containing the number of neurons in each hidden layer.
        - lr: The learning rate for the DQN.
        - gamma: The discount factor for future rewards.
    """
    def __init__(self, state_dim, action_dim, hidden_dims, lr=0.05, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.lr = lr
        self.gamma = gamma

        # Neural Network
        self.weights = []
        previous_dim = state_dim

        for hidden_dim in hidden_dims:
            self.weights.append([[random.uniform(-1, 1) for _ in range(hidden_dim)] for _ in range(previous_dim)])
            previous_dim = hidden_dim

        # Final layer should have dimensionality matching action space
        self.weights.append([[random.uniform(-1, 1) for _ in range(action_dim)] for _ in range(previous_dim)])

    def forward(self, state):
        """
        Forward pass through the network. Accepts a state as input and returns the action values as output.
        """
        self.a = []
        self.z = []
    
        if isinstance(state, int):
            state_list = [0]*self.state_dim
            state_list[state] = 1
            state = state_list
    
        if not isinstance(state, list):  # Ensure state is always a list
            state = [state]
            
        self.a.append(state)
    
        for i in range(len(self.weights)):
            self.z.append(self.dot(self.a[-1], self.weights[i]))
            self.a.append([self.relu(x) for x in self.z[-1]])
    
        return self.a[-1][0:self.action_dim]
    
    def backward(self, state, action, target):
        """
        Backward pass through the network. Accepts a state, action, and target as inputs and updates the weights of the network.
        """
        if not isinstance(state, list):  # Ensure state is always a list
            state = [state]
    
        self.a = [state, ] + self.a
        deltas = [[0]*len(layer) for layer in self.weights]  # Initialize deltas
    
        # Compute output layer delta
        deltas[-1] = [0]*len(self.weights[-1])  # Initialize last layer delta
        deltas[-1][action] = target - self.a[-1][action]
    
        # Backpropagate deltas
        for i in reversed(range(len(self.weights) - 1)):  # Start from the end, excluding the last layer
            for k in range(len(self.weights[i+1])):  # For each neuron in the next layer
                for j in range(len(self.weights[i+1][k])):  # For each neuron in the current layer
                    if j < len(self.a[i+1]):
                        deltas[i][j] += deltas[i+1][k] * self.weights[i+1][k][j] * (self.a[i+1][j] > 0)  # Apply ReLU derivative
    
        # Update weights
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    if j < len(self.a[i]) and k < len(deltas[i]):
                        self.weights[i][j][k] += self.lr * self.a[i][j] * deltas[i][k]  # Update weights using the deltas
    
    
    def update(self, state, action, reward, next_state):
        """
        Update the network given a state, action, reward, and the next state.
        """
        forward_state = self.forward(state)
        future_rewards = max(self.forward(next_state))
        target = reward + self.lr * (reward + self.gamma * future_rewards - forward_state[action])
        self.backward(state, action, target)


    def get_action(self, state, epsilon):
        """
        Select an action for a given state using epsilon-greedy action selection.
        """
        forward_state = self.forward(state)
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            return forward_state.index(max(forward_state))
            
            
    def dot(self, a, b):
        """
        Dot product operation.
        """
        # If 'a' and 'b' are 1D lists
        if not any(isinstance(i, list) for i in a) and not any(isinstance(i, list) for i in b):
                return sum(x*y for x, y in zip(a, b))
        
        # If 'a' and 'b' are 2D lists
        elif all(isinstance(i, list) for i in a) and all(isinstance(i, list) for i in b):
            b_t = list(map(list, zip(*b))) # Transpose b
            return [[sum(x*y for x, y in zip(row_a, row_b)) for row_b in b_t] for row_a in a]
        
        # 'b' is a 2D list, 'a' is a 1D list
        elif all(isinstance(i, list) for i in b):
            return [sum(x*y for x, y in zip(a, neuron_weights)) for neuron_weights in b]
        
        # 'b' is a 1D list, 'a' is a 1D list
        else:
            return sum(x*y for x, y in zip(a, b))
            
    def relu(self, x):
        """
        ReLU (Rectified Linear Unit) activation function.
        """
        return max(0, x)
    
    def outer(self, a, b):
        """
        Outer product operation.
        """
        if isinstance(b, float):
            return [[x*b for _ in range(len(a))] for x in a]
        else:
            return [[x*y for y in b] for x in a]
    
    def sigmoid(self, x):
        """
        Sigmoid activation function.
        """
        return [1 / (1 + math.exp(-xi)) for xi in x]


class QLearning:
    """
        Initialize the Q-Learning object with the following parameters:
        - states: The total number of states.
        - actions: The total number of actions.
        - lr: The learning rate for Q-Learning.
        - gamma: The discount factor for future rewards.
    """
    def __init__(self, states, actions, lr=0.05, gamma=0.99):
        self.states = states
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        # Initialize Q-table with zeros
        self.q_table = [[0 for _ in range(actions)] for _ in range(states)]

    def update(self, state, action, reward, next_state):
        """
        Update the Q-table given a state, action, reward, and the next state.
        """
        future_rewards = max(self.q_table[next_state])
        target = reward + self.gamma * future_rewards
        self.q_table[state][action] = (1 - self.lr) * self.q_table[state][action] + self.lr * target

    def get_action(self, state, epsilon):
        """
        Select an action for a given state using epsilon-greedy action selection.
        """
        if random.getrandbits(8) < 255 * epsilon:
            return random.getrandbits(8) % self.actions
        else:
            return self.q_table[state].index(max(self.q_table[state]))

# Define a pseudo-environment
class PseudoEnvironment:
    def __init__(self):
        """
        Initialize the PseudoEnvironment object.
        """
        self.current_state = 0 # Start with state 1

    def step(self, action):
        """
        Take a step in the environment given an action.
        """
        reward = 0 # Default reward
        if action == 0:
            self.current_state = 1 # Go to state 2
            reward = 1
        elif action == 1:
            self.current_state = 2 # Go to state 3
            reward = -1

        return self.current_state, reward
        
        
# Initialize DQN, Q-Learning and Environment
dqn = DQN(state_dim=3, action_dim=2, hidden_dims=[10,10], lr=0.5, gamma=0.95)
q_learning = QLearning(states=3, actions=2, lr=0.5, gamma=0.95)
env = PseudoEnvironment()

# Initialize lists to store total rewards per episode
rewards_dqn = []
rewards_q_learning = []

# Number of episodes for DQN
num_episodes = 1000

# Training loop for DQN
for i in range(num_episodes):
    state = env.current_state
    total_reward = 0
    done = False
    while not done:
        action = dqn.get_action(state, epsilon=0.1)
        next_state, reward = env.step(action)
        dqn.update(state, action, reward, next_state)
        total_reward += reward
        if next_state == 2: # Assuming reaching state 3 ends the episode
            done = True
        state = next_state
    rewards_dqn.append(total_reward)

# Number of episodes for Q-Learning
num_episodes = 1000

# Reset environment
env = PseudoEnvironment()

# Training loop for Q-Learning
for i in range(num_episodes):
    state = env.current_state
    total_reward = 0
    done = False
    while not done:
        action = q_learning.get_action(state, epsilon=0.1)
        next_state, reward = env.step(action)
        q_learning.update(state, action, reward, next_state)
        total_reward += reward
        if next_state == 2: # Assuming reaching state 3 ends the episode
            done = True
        state = next_state
    rewards_q_learning.append(total_reward)


# Save the rewards to a file
with open('rewards.json', 'w') as f:
    json.dump({'dqn': rewards_dqn, 'q_learning': rewards_q_learning}, f)