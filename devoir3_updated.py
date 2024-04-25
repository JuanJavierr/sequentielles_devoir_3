"""
Devoir 3: Décisions Séquentielles

Main functions: train_qlearning, train_dqn, test_policy_robot
"""

# import codey, event, rocky # uncomment this line when running on the robot
import time, random, math
import matplotlib.pyplot as plt

"""
CLASSES
"""

# Class definition for Q-Learning
# This class is directly taken from "DQN_Q-Learning.py".
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



# Class definition for Deep Q Network (DQN)
# This class is directly taken from "DQN_Q-Learning.py".
"""
NOTE: In this class, we add a variable q_table, so that we can store all the Q-values
for each state during training.

Normally, in DQN, we prefer returning the weights instead of the Q-values, but since our
environment is static and contains a small discrete state space, it would be better
to return the Q-values obtained through the "update" function in this class
instead of the weights.

We test out both of the methods (when we return the q-table,
vs when we return the weights) and we got the same Q-values.

Judging from this implementation, we realized that DQN could be better utilized in
more complex and dynamic environment.
"""
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
        self.q_table = [[0 for _ in range(action_dim)] for _ in range(state_dim)]

        # Neural Network
        self.weights = []
        previous_dim = state_dim

        for hidden_dim in hidden_dims:
            self.weights.append([[random.uniform(-1, 1) for _ in range(hidden_dim)] for _ in range(previous_dim)])
            previous_dim = hidden_dim

        # Final layer should have dimensionality matching action space
        self.weights.append([[random.uniform(-1, 1) for _ in range(action_dim)] for _ in range(previous_dim)])


        self.target_weights = self.weights.copy()

    def forward(self, state, use_target=False):
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


        if use_target:
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
        self.q_table[state] = forward_state # new line added: update the q-values for the specific state
        future_rewards = max(self.forward(next_state, use_target=True))

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



"""
IMPLEMENTED FUNCTIONS
"""

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
    print('Action taken: ' + str(action))
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

    # 0: Up
    # 1: Right
    # 2: Down
    # 3: Left
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



def train_qlearning():
  "Training stage of the Robot in the Maze using Q-Learning"

  global q_learning

  num_episodes = 200
  total_rewards = []
  steps = []

  for _ in range(num_episodes):
      total_reward = 0
      last_action = 0 # initial action is up
      state = (4, 3) # starting state
      is_done = False
      number_steps = 0

      while not is_done: # while maze is not solved
          # Decide on the action using an epsilon-greedy method
          epsilon = 0.7/(_ + 1)
          action = q_learning.get_action(convert_state_to_index(state), epsilon=epsilon)
          next_state, reward, is_done = take_action(state, action)
          total_reward += reward

          q_learning.update(
               convert_state_to_index(state),
               action,
               reward,
               convert_state_to_index(next_state),
          )

          state = next_state
          last_action = action
          number_steps += 1

          if is_done:
              print("WoW!, episode:", _, "reward:", total_reward, "steps:", number_steps)
              total_rewards.append(total_reward)
              steps.append(number_steps)

  plt.title("Total Rewards in Episodes (Q-Learning)")
  plt.plot(range(0,num_episodes), total_rewards, '-')
  plt.show()

  plt.title("Number of Steps in Episodes (Q-Learning)")
  plt.plot(range(0,num_episodes), steps, '-go')
  plt.show()

  return q_learning.q_table



def train_dqn():
  "Training stage of the Robot in the Maze using DQN"
  global dqn

  num_episodes = 200
  total_rewards = []
  steps = []

  for _ in range(num_episodes):
    total_reward = 0
    last_action = 0
    state = (4,3)
    is_done = False
    number_steps = 0

    while not is_done:
      epsilon = 0.3
      action = dqn.get_action(convert_state_to_index(state), epsilon=epsilon)
      next_state, reward, is_done = take_action(state, action)
      total_reward += reward

      dqn.update(
          convert_state_to_index(state),
          action,
          reward,
          convert_state_to_index(next_state)
      )

      state = next_state
      last_action = action
      number_steps += 1

      if is_done:
        print("WoW!, episode:", _, "reward:", total_reward, "steps:", number_steps)
        total_rewards.append(total_reward)
        steps.append(number_steps)
        dqn.target_weights = dqn.weights.copy() # on episode end, update target weights

      if number_steps % 50 == 0: # update target weights every 50 steps
        dqn.target_weights = dqn.weights.copy()

  plt.title("Total Rewards in Episodes (DQN)")
  plt.plot(range(0,num_episodes), total_rewards, '-')
  plt.show()

  plt.title("Number of Steps in Episodes (DQN)")
  plt.plot(range(0,num_episodes), steps, '-go')
  plt.show()

  return dqn.q_table



def test_policy_robot(q_table):
  """
  NOTE: Since DQN gives us a final policy that lets the robot loop infinitely in the maze,
  we decide to display only 10 iterations, since the number of optimal actions to
  reach the exit is 8.
  """
  total_reward = 0
  last_action = 0 # initial action is up
  state = (4, 3) # starting state
  is_done = False
  iterations = 0
  number_steps = 0

  while not is_done and iterations < 10:
    action = q_table[convert_state_to_index(state)].index(
        max(q_table[convert_state_to_index(state)]))

    move_robot(last_action, action)
    next_state, reward, is_done = take_action(state, action)
    total_reward += reward

    state = next_state
    last_action = action
    number_steps += 1

    iterations += 1

    if is_done:
      print("The robot has successfully reached the exit!")
      rocky.stop()
      codey.display.show("WoW!")
      codey.led.set_green(255)
      for sound in ("yeah", "wow", "laugh"):
          codey.speaker.play_melody(sound, True)

    if iterations == 10 and not is_done:
      print("The robot is stuck in the maze.")
      rocky.stop()

"""
TRAINING AND TEST STAGE OF THE ROBOT
"""

# Q-Learning
random.seed(0)
q_learning = QLearning(states=5 * 4, actions=4, lr=0.2, gamma=0.9)
q_table = train_qlearning()

# @event.button_a_pressed
def on_button_a_pressed():
  test_policy_robot(q_table)

# Deep Q-Learning
random.seed(21)
dqn = DQN(state_dim=5*4, action_dim=4, hidden_dims=[25], lr=0.05, gamma=0.99)
q_table_dqn = train_dqn()

# @event.button_b_pressed
def on_button_b_pressed():
  test_policy_robot(q_table_dqn)