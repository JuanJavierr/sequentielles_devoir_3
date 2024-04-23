""""
Devoir 3 : Décisions Séquentielles

J'ai fait un gros fichier parce que la version de python sur le robot ne supporte pas les imports de modules.
Main functions: solve_maze_qlearning, solve_maze_dqn
"""
# import codey, event, rocky # uncomment this line when running on the robot
import time, random, math
from collections import deque

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

        self.target_weights = self.weights.copy()

    def forward(self, state, target=False):
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

        if target:
            weights = self.target_weights
        else:
            weights = self.weights
    
        for i in range(len(weights)):
            self.z.append(self.dot(self.a[-1], weights[i]))
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

    # def update(self, replay_buffer, batch_size = 100):
    #     if len(replay_buffer) < batch_size:
    #         return
    #     random_sample = random.sample(replay_buffer, batch_size)

    #     states = [x[0] for x in random_sample]
    #     actions = [x[1] for x in random_sample]
    #     rewards = [x[2] for x in random_sample]
    #     next_states = [x[3] for x in random_sample]
    #     dones = [x[4] for x in random_sample]

    #     q_values = self.forward(states)

    #     q_values


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
            # output = []
            # i = 0
            # for neuron_weights in b:
            #     i += 1
            #     summation = 0
            #     for x, y in zip(a, neuron_weights):
            #         summation += x*y
            #     output.append(summation)
            # return output
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

# Class definition for Q-Learning
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
        self.q_table[state][action] = (1 - self.lr) * self.q_table[state][
            action
        ] + self.lr * target

    def get_action(self, state, epsilon):
        """
        Select an action for a given state using epsilon-greedy action selection.
        """
        if random.getrandbits(8) < 255 * epsilon:
            return random.getrandbits(8) % self.actions
        else:
            return self.q_table[state].index(max(self.q_table[state]))


def move_robot(last_action: int, action: int):
    """Physically change the position of the robot based on the last and current action."""

    angle = (action - last_action) * 90

    if angle == 270: # avoid turning right 3 times
        rocky.turn_left_by_degree(90)
    elif angle == -270: # avoid turning left 3 times
        rocky.turn_right_by_degree(90)
    else:
        rocky.turn_right_by_degree(angle)
    last_action = action

    codey.display.show(action)
    rocky.forward(8, 1)
    time.sleep(1)

def get_new_state(state: tuple[int], action: int) -> tuple[int]:
    """Given a state and an action, return the new state."""

    if action == 0: #up
        return (state[0] - 1, state[1])
    if action == 2: #down
        return (state[0] + 1, state[1])
    if action == 3: #left
        return (state[0], state[1] - 1)
    
    return (state[0], state[1] + 1) #right


def get_available_directions(state: tuple[int]) -> set[int]:
    """Given a position in the maze, return the available directions to move."""

    orients = {0, 2, 3, 1}
    state = (state[0] - 1, state[1] - 1)  # convert to 0-based index

    MAZE_HWALLS = [ # horizontal walls
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ]


    MAZE_VWALLS = [ # vertical walls
        [1, 0, 0, 1, 0, 1],
        [1, 1, 1, 0, 1, 1],
        [1, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 1],
    ]


    bad_moves = {
        0: MAZE_HWALLS[state[0]][state[1]], # up
        1: MAZE_VWALLS[state[0]][state[1] + 1], # right
        2: MAZE_HWALLS[state[0] + 1][state[1]], # down
        3: MAZE_VWALLS[state[0]][state[1]], # left
    }

    for orient in orients.copy():
        if bad_moves[orient]:
            orients.remove(orient)

    return orients


def take_action(state: tuple[int], action: int) -> tuple[tuple[int], int, bool]:
    """Take an action in the environment and return the new state, reward, and done flag."""

    orients = get_available_directions(state)

    if action not in orients: # if action runs into a wall
        return state, -1, False
    
    if state == (1, 3) and action == 0: # if action reaches the goal
        return state, 100, True
    new_state = get_new_state(state, action)
    
    return new_state, 0, False # if action is not wall or goal

def convert_state_to_index(state: tuple[int]) -> int:
    """Convert 2-D state to 1-D index."""
    return (5 * (state[0] - 1) + (state[1] - 1))

def solve_maze_qlearning(is_training=False):
    """
    Train the agent using Q-Learning.
    If cody is True, 
    """
    global q_learning

    if is_training:
        num_episodes = 300
    else:
        num_episodes = 1

    for _ in range(num_episodes):
        total_reward = 0
        last_action = 0 # initial action is up
        state = (4, 3) # starting state
        is_done = False

        while not is_done: # while maze is not solved
            # Decide on the action using an epsilon-greedy method
            if is_training:
                epsilon = 0.7/(_ + 1)
            else:
                epsilon = 0
            action = q_learning.get_action(convert_state_to_index(state), epsilon=epsilon)

            if not is_training:
                move_robot(last_action, action)

            next_state, reward, is_done = take_action(state, action, is_training)
            total_reward += reward

            q_learning.update(
                convert_state_to_index(state),
                action,
                reward,
                convert_state_to_index(next_state),
            )

            
            state = next_state
            last_action = action


            if is_done:
                if is_training:
                    print("WoW!, episode:", _, "reward:", total_reward)
                else:
                    rocky.stop()
                    codey.display.show("WoW!")
                    codey.led.set_green(255)
                    for sound in ("yeah", "wow", "laugh"):
                        codey.speaker.play_melody(sound, True)

def solve_maze_dqn(is_training=False):
    global dqn

    if is_training:
        num_episodes = 2000
    else:
        num_episodes = 1
    total_steps = 0

    total_rewards = []
    for i in range(num_episodes):
        episode_reward = 0
        last_action = 0
        state = (4, 3)
        is_done = False
        steps = 0

        while not is_done:
            if is_training:
                epsilon = 0.3
            else:
                epsilon = 0
            action = dqn.get_action(convert_state_to_index(state), epsilon=epsilon)

            if not is_training:
                move_robot(last_action, action)

            next_state, reward, is_done = take_action(state, action)
            episode_reward += reward

            dqn.update(
                convert_state_to_index(state),
                action,
                reward,
                convert_state_to_index(next_state)
            )

            state = next_state
            last_action = action
            steps += 1
            total_steps += 1

            if is_done:
                if is_training:
                    print("WoW!, episode:", i, "reward:", episode_reward, "steps:", steps)
                    total_rewards.append(episode_reward)
                    # dqn.target_weights = dqn.weights.copy()
                else:
                    rocky.stop()
                    codey.display.show("WoW!")
                    codey.led.set_green(255)
                    for sound in ("yeah", "wow", "laugh"):
                        codey.speaker.play_melody(sound, True)


            if is_training and total_steps % 1000 == 0:
                dqn.target_weights = dqn.weights.copy()
                print("Updated target weights")

    return total_rewards




q_learning = QLearning(states=5 * 4, actions=4, lr=0.2, gamma=0.9)
dqn = DQN(state_dim=5*4, action_dim=4, hidden_dims=[10, 10], lr=0.05, gamma=0.99)
# @event.button_a_pressed
def on_button_a_pressed():
    solve_maze_qlearning(is_training=False)



# @event.button_b_pressed
def on_button_b_pressed():
    # codey.display.clear()
    # solve_maze_qlearning(is_training=True)
    return solve_maze_dqn(is_training=True)
    # codey.speaker.volume = 100
    # codey.speaker.play_melody("hello", True)

total_rewards = on_button_b_pressed()
# %%
import matplotlib.pyplot as plt

plt.plot(total_rewards)
